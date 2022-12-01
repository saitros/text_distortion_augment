# Import modules
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import gc
import h5py
import pickle
import psutil
import logging
import random
import datetime
import numpy as np
from tqdm import tqdm
from time import time
# Import PyTorch
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from transformers import BertForSequenceClassification
# Import custom modules
from model.model import AugModel
from model.dataset import CustomDataset
from model.loss import MaximumMeanDiscrepancy
from optimizer.utils import shceduler_select, optimizer_select
from utils import TqdmLoggingHandler, write_log, get_tb_exp_name

def training(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #===================================#
    #==============Logging==============#
    #===================================#

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False

    write_log(logger, 'Start training!')

    #===================================#
    #============Data Load==============#
    #===================================#

    # 1) Data open
    write_log(logger, "Load data...")
    gc.disable()

    save_path = os.path.join(args.preprocess_path, args.data_name)

    with h5py.File(os.path.join(save_path, 'processed.hdf5'), 'r') as f:
        train_src_input_ids = f.get('train_src_input_ids')[:]
        train_src_attention_mask = f.get('train_src_attention_mask')[:]
        train_src_token_type_ids = f.get('train_src_token_type_ids')[:]
        valid_src_input_ids = f.get('valid_src_input_ids')[:]
        valid_src_attention_mask = f.get('valid_src_attention_mask')[:]
        valid_src_token_type_ids = f.get('valid_src_token_type_ids')[:]
        train_trg_list = f.get('train_label')[:]
        train_trg_list = F.one_hot(torch.tensor(train_trg_list, dtype=torch.long)).numpy()
        valid_trg_list = f.get('valid_label')[:]
        valid_trg_list = F.one_hot(torch.tensor(valid_trg_list, dtype=torch.long)).numpy()

    with open(os.path.join(save_path, 'word2id.pkl'), 'rb') as f:
        data_ = pickle.load(f)
        src_word2id = data_['src_word2id']
        src_vocab_num = len(src_word2id)
        num_labels = data_['num_labels']
        del data_

    gc.enable()
    write_log(logger, "Finished loading data!")

    #===================================#
    #===========Train setting===========#
    #===================================#

    # 1) Model initiating
    write_log(logger, 'Instantiating model...')
    model = AugModel(encoder_model_type='bert', decoder_model_type='bert',
                     isPreTrain=args.isPreTrain, z_variation=args.z_variation,
                     src_max_len=args.src_max_len, dropouot=args.dropouot)
    model.to(device)

    # 2) Dataloader setting
    dataset_dict = {
        'train': CustomDataset(src_list=train_src_input_ids, src_att_list=train_src_attention_mask,
                               src_seg_list=train_src_token_type_ids,
                               trg_list=train_trg_list, src_max_len=args.src_max_len),
        'valid': CustomDataset(src_list=valid_src_input_ids, src_att_list=valid_src_attention_mask,
                               src_seg_list=valid_src_token_type_ids,
                               trg_list=valid_trg_list, src_max_len=args.src_max_len),
    }
    dataloader_dict = {
        'train': DataLoader(dataset_dict['train'], drop_last=True,
                            batch_size=args.batch_size, shuffle=True, pin_memory=True,
                            num_workers=args.num_workers),
        'valid': DataLoader(dataset_dict['valid'], drop_last=False,
                            batch_size=args.batch_size, shuffle=False, pin_memory=True,
                            num_workers=args.num_workers)
    }
    write_log(logger, f"Total number of trainingsets  iterations - {len(dataset_dict['train_original'])}, {len(dataloader_dict['train_original'])}")
    
    # 3) Optimizer & Learning rate scheduler setting
    optimizer = optimizer_select(model, args)
    scheduler = shceduler_select(model, dataloader_dict, args)
    scaler = GradScaler()
    recon_loss = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing_eps)

    # 3) Model resume
    start_epoch = 0
    if args.resume:
        write_log(logger, 'Resume model...')
        save_path = os.path.join(args.model_save_path, args.task, args.data_name, args.tokenizer)
        save_file_name = os.path.join(save_path, 
                                        f'checkpoint_src_{args.src_vocab_size}_trg_{args.trg_vocab_size}_v_{args.variational_mode}_p_{args.parallel}.pth.tar')
        checkpoint = torch.load(save_file_name)
        start_epoch = checkpoint['epoch'] - 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        del checkpoint

    #===================================#
    #=========Model Train Start=========#
    #===================================#

    write_log(logger, 'Traing start!')
    best_val_loss = 1e+10
    
    for epoch in range(start_epoch + 1, args.num_epochs + 1):
        start_time_e = time()
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                write_log(logger, 'Validation start...')
                val_loss = 0
                val_acc = 0
                model.eval()

            for i, batch_iter in enumerate(tqdm(dataloader_dict[phase], bar_format='{l_bar}{bar:30}{r_bar}{bar:-2b}')):

                # Optimizer setting
                optimizer.zero_grad(set_to_none=True)

                # Input setting
                src_sequence = batch_iter[0]
                src_att = batch_iter[1]
                src_seg = batch_iter[2]
                trg_label = batch_iter[3]

                src_sequence = src_sequence.to(device, non_blocking=True)
                src_att = src_att.to(device, non_blocking=True)
                src_seg = src_seg.to(device, non_blocking=True)
                trg_label = trg_label.to(device, non_blocking=True)

                # Train
                if phase == 'train':
                    with autocast():
                        encoder_out, decoder_out, z = model(src_input_ids=src_sequence, 
                                                            src_attention_mask=src_att,
                                                            src_token_type_ids=src_seg)
                        mmd_loss = MaximumMeanDiscrepancy(encoder_out, z)
                        ce_loss = recon_loss()

                    scaler.scale(loss).backward()
                    if args.clip_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    # loss.backward()
                    # clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                    # optimizer.step()

                    if args.scheduler in ['constant', 'warmup']:
                        scheduler.step()
                    if args.scheduler == 'reduce_train':
                        scheduler.step(loss)

                    # Print loss value only training
                    if i == 0 or freq == args.print_freq or i==len(dataloader_dict[phase]):
                        acc = (predicted.max(dim=1)[1] == trg_label.argmax(dim=1)).sum() / len(trg_label)
                        iter_log = "[Epoch:%03d][%03d/%03d] train_loss:%03.2f | train_acc:%03.2f%% | learning_rate:%1.6f | spend_time:%02.2fmin" % \
                            (epoch, i, len(dataloader_dict[phase]), loss, acc*100, optimizer.param_groups[0]['lr'], (time() - start_time_e) / 60)
                        write_log(logger, iter_log)
                        freq = 0
                    freq += 1

                # Validation
                if phase == 'valid':
                    with torch.no_grad():
                        predicted = model(input_ids=src_sequence, attention_mask=src_att)['logits']
                        loss = F.cross_entropy(predicted, trg_label.argmax(dim=1))
                    val_loss += loss
                    val_acc += (predicted.max(dim=1)[1] == trg_label.argmax(dim=1)).sum() / len(trg_label.argmax(dim=1))
                    
            if phase == 'valid':

                if args.scheduler == 'reduce_valid':
                    scheduler.step(val_loss)
                if args.scheduler == 'lambda':
                    scheduler.step()

                val_loss /= len(dataloader_dict[loader_phase])
                val_acc /= len(dataloader_dict[loader_phase])
                write_log(logger, f'Mode: {total_phase}')
                write_log(logger, 'Validation Loss: %3.3f' % val_loss)
                write_log(logger, 'Validation Accuracy: %3.2f%%' % (val_acc * 100))

                save_file_name = os.path.join(args.model_save_path, args.data_name, total_phase.split('_')[-1])
                save_file_name += 'checkpoint2.pth.tar'
                if val_loss < best_val_loss:
                    write_log(logger, 'Checkpoint saving...')
                    torch.save({
                        'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'scaler': scaler.state_dict()
                    }, save_file_name)
                    best_val_loss = val_loss
                    best_epoch = epoch
                else:
                    else_log = f'Still {best_epoch} epoch Loss({round(best_val_loss.item(), 2)}) is better...'
                    write_log(logger, else_log)

                with open(f'./results_{time_code}.txt', 'a') as f:
                    f.write(f'{args.data_name}')
                    f.write('\t')
                    f.write(f'{total_phase}')
                    f.write('\t')
                    f.write(f'{epoch}')
                    f.write('\t')
                    f.write(f'{val_loss}')
                    f.write('\t')
                    f.write(f'{val_acc}')
                    f.write('\n')

    # 3) Results
    write_log(logger, f'Best Epoch: {best_epoch}')
    write_log(logger, f'Best Loss: {round(best_val_loss.item(), 2)}')