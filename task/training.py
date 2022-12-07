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
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
# Import custom modules
from model.model import AugModel, ClsModel
from model.dataset import CustomDataset
from model.loss import MaximumMeanDiscrepancy
from optimizer.utils import shceduler_select, optimizer_select
from utils import TqdmLoggingHandler, write_log, get_tb_exp_name
from task.utils import input_to_device

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

    with h5py.File(os.path.join(save_path, args.cls_tokenizer, 'processed.hdf5'), 'r') as f:
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
    
    with h5py.File(os.path.join(save_path, args.aug_tokenizer, 'processed.hdf5'), 'r') as f:
        aug_train_src_input_ids = f.get('train_src_input_ids')[:]
        aug_train_src_attention_mask = f.get('train_src_attention_mask')[:]
        aug_train_src_token_type_ids = f.get('train_src_token_type_ids')[:]
        aug_valid_src_input_ids = f.get('valid_src_input_ids')[:]
        aug_valid_src_attention_mask = f.get('valid_src_attention_mask')[:]
        aug_valid_src_token_type_ids = f.get('valid_src_token_type_ids')[:]
        aug_train_trg_list = f.get('train_label')[:]
        aug_train_trg_list = F.one_hot(torch.tensor(train_trg_list, dtype=torch.long)).numpy()
        aug_valid_trg_list = f.get('valid_label')[:]
        aug_valid_trg_list = F.one_hot(torch.tensor(valid_trg_list, dtype=torch.long)).numpy()

    with open(os.path.join(save_path, args.cls_tokenizer, 'word2id.pkl'), 'rb') as f:
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
    aug_model = AugModel(encoder_model_type='bert', decoder_model_type='bert',
                         isPreTrain=args.isPreTrain, z_variation=args.z_variation,
                         dropout=args.dropout)
    cls_model = ClsModel(model_type=args.cls_model_type, num_labels=num_labels)

    aug_model.to(device)
    cls_model.to(device)

    # 2) Dataloader setting
    dataset_dict = {
        'train': CustomDataset(src_list=train_src_input_ids, src_att_list=train_src_attention_mask,
                               src_seg_list=train_src_token_type_ids,
                               trg_list=train_trg_list, src_max_len=args.src_max_len),
        'valid': CustomDataset(src_list=valid_src_input_ids, src_att_list=valid_src_attention_mask,
                               src_seg_list=valid_src_token_type_ids,
                               trg_list=valid_trg_list, src_max_len=args.src_max_len),
        'aug_train': CustomDataset(src_list=aug_train_src_input_ids, src_att_list=aug_train_src_attention_mask,
                                   src_seg_list=aug_train_src_token_type_ids,
                                   trg_list=aug_train_trg_list, src_max_len=args.src_max_len),
        'aug_valid': CustomDataset(src_list=aug_valid_src_input_ids, src_att_list=aug_valid_src_attention_mask,
                                   src_seg_list=aug_valid_src_token_type_ids,
                                   trg_list=aug_valid_trg_list, src_max_len=args.src_max_len),
    }
    dataloader_dict = {
        'train': DataLoader(dataset_dict['train'], drop_last=False,
                            batch_size=args.batch_size, shuffle=False, pin_memory=True,
                            num_workers=args.num_workers),
        'valid': DataLoader(dataset_dict['valid'], drop_last=False,
                            batch_size=args.batch_size, shuffle=False, pin_memory=True,
                            num_workers=args.num_workers),
        'aug_train': DataLoader(dataset_dict['aug_train'], drop_last=False,
                            batch_size=args.batch_size, shuffle=False, pin_memory=True,
                            num_workers=args.num_workers),
        'aug_valid': DataLoader(dataset_dict['aug_valid'], drop_last=False,
                            batch_size=args.batch_size, shuffle=False, pin_memory=True,
                            num_workers=args.num_workers)
    }
    write_log(logger, f"Total number of trainingsets  iterations - {len(dataset_dict['train'])}, {len(dataloader_dict['train'])}")

    del (
        train_src_input_ids, train_src_attention_mask, train_src_token_type_ids, train_trg_list,
        valid_src_input_ids, valid_src_attention_mask, valid_src_token_type_ids, valid_trg_list
    )
    
    # 3) Optimizer & Learning rate scheduler setting
    optimizer = optimizer_select(aug_model, args)
    scheduler = shceduler_select(optimizer, dataloader_dict, args)
    cudnn.benchmark = True
    scaler = GradScaler()
    cls_loss = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing_eps).to(device)
    recon_loss = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing_eps, ignore_index=aug_model.pad_idx).to(device)

    # 3) Model resume
    start_epoch = 0
    if args.resume:
        write_log(logger, 'Resume model...')
        save_path = os.path.join(args.model_save_path, args.task, args.data_name, args.tokenizer)
        save_file_name = os.path.join(save_path, 
                                        f'checkpoint_src_{args.src_vocab_size}_trg_{args.trg_vocab_size}_v_{args.variational_mode}_p_{args.parallel}.pth.tar')
        checkpoint = torch.load(save_file_name)
        start_epoch = checkpoint['epoch'] - 1
        aug_model.load_state_dict(checkpoint['aug_model'])
        cls_model.load_state_dict(checkpoint['cls_model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        del checkpoint

    #===================================#
    #=========Model Train Start=========#
    #===================================#

    write_log(logger, 'Traing start!')
    best_val_loss = 1e+3
    
    for epoch in range(start_epoch + 1, args.num_epochs + 1):
        while True:
            start_time_e = time()
            finish_epoch = False
            for phase in ['cls_train', 'cls_valid', 'aug_train', 'aug_valid']:

                if 'train' in phase:
                    cls_model.train()
                    aug_model.train()
                if 'valid' in phase:
                    cls_model.eval()
                    aug_model.eval()

                # 1. Classifier training
                # Optimizer setting
                optimizer.zero_grad(set_to_none=True)

                for _ in range(args.num_grad_accumulate):

                    try:
                        batch_iter = next(dataloader_dict['train'])
                        aug_batch_iter = next(dataloader_dict['aug_train'])
                    except StopIteration:
                        batch_iter = iter(dataloader_dict['train'])
                        aug_batch_iter = iter(dataloader_dict['aug_train'])
                        batch_iter = next(dataloader_dict['train'])
                        aug_batch_iter = next(dataloader_dict['aug_train'])
                        finish_epoch = True

                    # Input setting
                    b_iter, aug_b_iter = input_to_device(batch_iter, aug_batch_iter, device)
                    src_sequence, src_att, src_seg, trg_label = b_iter
                    aug_src_sequence, aug_src_att, aug_src_seg, aug_trg_label = aug_b_iter

                    # Reconsturction setting
                    trg_sequence_gold = src_sequence.contiguous().view(-1)
                    non_pad = trg_sequence_gold != aug_model.pad_idx

                    # Classifier training
                    with autocast():
                        logit = cls_model(src_input_ids=src_sequence,
                                          src_attention_mask=src_att)
                        cls_loss = recon_loss(logit, trg_label)/args.num_grad_accumulate

                    scaler.scale(cls_loss).backward()
                    
                scaler.step(optimizer)
                scaler.update()

                    # Augmenter training
                    with autocast():
                        encoder_out, decoder_out, z = aug_model(src_input_ids=src_sequence, 
                                                                src_attention_mask=src_att,
                                                                src_token_type_ids=src_seg)
                        mmd_loss = MaximumMeanDiscrepancy(encoder_out.view(args.batch_size, -1), 
                                                          z.view(args.batch_size, -1), 
                                                          z_var=args.z_variation) * 10
                        ce_loss = recon_loss(decoder_out.view(-1, src_vocab_num), trg_sequence_gold)
                        total_loss = mmd_loss + ce_loss

                    scaler.scale(total_loss).backward()
                    if args.clip_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        clip_grad_norm_(aug_model.parameters(), args.clip_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()

                    if args.scheduler in ['constant', 'warmup']:
                        scheduler.step()
                    if args.scheduler == 'reduce_train':
                        scheduler.step(total_loss)

                    # Print loss value only training
                    if i == 0 or freq == args.print_freq or i==(len(dataloader_dict[phase])-1):
                        acc = (decoder_out.argmax(dim=2).view(-1)[non_pad] == trg_sequence_gold[non_pad]).sum() / len(trg_sequence_gold[non_pad])
                        iter_log = "[Epoch:%03d][%03d/%03d] train_ce_loss:%03.2f | train_mmd_loss:%03.2f | train_acc:%03.2f%% | learning_rate:%1.6f | spend_time:%02.2fmin" % \
                            (epoch, i+1, len(dataloader_dict[phase]), ce_loss.item(), mmd_loss.item(), acc*100, optimizer.param_groups[0]['lr'], (time() - start_time_e) / 60)
                        write_log(logger, iter_log)
                        freq = 0
                    freq += 1

                else:
                    write_log(logger, 'Validation start...')
                    val_mmd_loss = 0
                    val_ce_loss = 0
                    val_acc = 0
                    aug_model.eval()
                    cls_model.train()

                    # Validation
                    if phase == 'valid':
                        with torch.no_grad():
                            encoder_out, decoder_out, z = model(src_input_ids=src_sequence, 
                                                                src_attention_mask=src_att,
                                                                src_token_type_ids=src_seg)
                            mmd_loss = MaximumMeanDiscrepancy(encoder_out.view(args.batch_size, -1), 
                                                            z.view(args.batch_size, -1), 
                                                            z_var=args.z_variation) * 10
                            ce_loss = F.cross_entropy(decoder_out.view(-1, src_vocab_num), trg_sequence_gold)
                        val_mmd_loss += mmd_loss
                        val_ce_loss += ce_loss
                        val_acc += (decoder_out.argmax(dim=2).view(-1)[non_pad] == trg_sequence_gold[non_pad]).sum() / len(trg_sequence_gold[non_pad])
                        
                if phase == 'valid':

                    if args.scheduler == 'reduce_valid':
                        scheduler.step(val_ce_loss)
                    if args.scheduler == 'lambda':
                        scheduler.step()

                    val_ce_loss /= len(dataloader_dict[phase])
                    val_mmd_loss /= len(dataloader_dict[phase])
                    val_acc /= len(dataloader_dict[phase])
                    write_log(logger, 'Validation CrossEntropy Loss: %3.3f' % val_ce_loss)
                    write_log(logger, 'Validation MMD Loss: %3.3f' % val_mmd_loss)
                    write_log(logger, 'Validation Accuracy: %3.2f%%' % (val_acc * 100))

                    save_file_name = os.path.join(args.model_save_path, args.data_name)
                    save_file_name += 'checkpoint.pth.tar'
                    if val_ce_loss < best_val_loss:
                        write_log(logger, 'Checkpoint saving...')
                        torch.save({
                            'epoch': epoch,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'scaler': scaler.state_dict()
                        }, save_file_name)
                        best_val_loss = val_ce_loss
                        best_epoch = epoch
                    else:
                        else_log = f'Still {best_epoch} epoch Loss({round(best_val_loss.item(), 2)}) is better...'
                        write_log(logger, else_log)

    # 3) Results
    write_log(logger, f'Best Epoch: {best_epoch}')
    write_log(logger, f'Best Loss: {round(best_val_loss.item(), 2)}')