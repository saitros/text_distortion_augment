# Import modules
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import gc
import h5py
import pickle
import logging
import datetime
import numpy as np
from tqdm import tqdm
from time import time
# Import PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# Import custom modules
from model.model import TransformerModel, ClassifierModel
from model.dataset import CustomDataset
from model.loss import compute_mmd, CustomLoss
from optimizer.utils import shceduler_select, optimizer_select
from optimizer.scheduler import get_cosine_schedule_with_warmup
from utils import TqdmLoggingHandler, write_log, get_tb_exp_name
from task.utils import input_to_device

def augmenter_training2(args):
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

    save_path = os.path.join(args.preprocess_path, args.data_name, args.model_type)

    with h5py.File(os.path.join(save_path, 'processed.hdf5'), 'r') as f:
        train_src_input_ids = f.get('train_src_input_ids')[:]
        train_src_attention_mask = f.get('train_src_attention_mask')[:]
        valid_src_input_ids = f.get('valid_src_input_ids')[:]
        valid_src_attention_mask = f.get('valid_src_attention_mask')[:]
        train_trg_list = f.get('train_label')[:]
        train_trg_list = F.one_hot(torch.tensor(train_trg_list, dtype=torch.long)).numpy()
        valid_trg_list = f.get('valid_label')[:]
        valid_trg_list = F.one_hot(torch.tensor(valid_trg_list, dtype=torch.long)).numpy()
        if args.model_type == 'bert':
            train_src_token_type_ids = f.get('train_src_token_type_ids')[:]
            valid_src_token_type_ids = f.get('valid_src_token_type_ids')[:]
        else:
            train_src_token_type_ids = list()
            valid_src_token_type_ids = list()
        
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
    aug_model = TransformerModel(model_type=args.model_type, src_max_len=args.src_max_len,
                                 isPreTrain=args.isPreTrain, dropout=args.dropout,
                                 encoder_out_ratio=args.encoder_out_ratio)
    cls_model = ClassifierModel(d_latent=aug_model.d_hidden, num_labels=num_labels, dropout=args.dropout)
    aug_model.to(device)
    cls_model.to(device)

    # 2) Dataloader setting
    dataset_dict = {
        'train': CustomDataset(src_list=train_src_input_ids, src_att_list=train_src_attention_mask,
                               src_seg_list=train_src_token_type_ids,
                               trg_list=train_trg_list, src_max_len=args.src_max_len),
        'valid': CustomDataset(src_list=valid_src_input_ids, src_att_list=valid_src_attention_mask,
                               src_seg_list=valid_src_token_type_ids,
                               trg_list=valid_trg_list, src_max_len=args.src_max_len)
    }
    dataloader_dict = {
        'train': DataLoader(dataset_dict['train'], drop_last=False,
                            batch_size=args.batch_size, shuffle=True,
                            pin_memory=True, num_workers=args.num_workers),
        'valid': DataLoader(dataset_dict['valid'], drop_last=False,
                            batch_size=args.batch_size, shuffle=False, pin_memory=True,
                            num_workers=args.num_workers)
    }
    write_log(logger, f"Total number of trainingsets  iterations - {len(dataset_dict['train'])}, {len(dataloader_dict['train'])}")

    del (
        train_src_input_ids, train_src_attention_mask, train_src_token_type_ids, train_trg_list,
        valid_src_input_ids, valid_src_attention_mask, valid_src_token_type_ids, valid_trg_list
    )

    # 3) Optimizer & Learning rate scheduler setting
    cls_optimizer = optimizer_select(optimizer_model=args.cls_optimizer, model=aug_model, lr=args.cls_lr)
    aug_optimizer = optimizer_select(optimizer_model=args.aug_optimizer, model=aug_model, lr=args.aug_lr)
    cls_scheduler = shceduler_select(phase='cls', scheduler_model=args.cls_scheduler, optimizer=cls_optimizer, dataloader_len=len(dataloader_dict['train']), args=args)
    aug_scheduler = shceduler_select(phase='aug', scheduler_model=args.aug_scheduler, optimizer=aug_optimizer, dataloader_len=len(dataloader_dict['train']), args=args)

    cudnn.benchmark = True
    cls_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing_eps).to(device)
    recon_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing_eps, ignore_index=aug_model.pad_idx).to(device)
    # cls_criterion = nn.CrossEntropyLoss().to(device)
    # recon_criterion = nn.CrossEntropyLoss(ignore_index=aug_model.pad_idx).to(device)

    # 3) Model resume
    start_epoch = 0
    if args.resume:
        write_log(logger, 'Resume model...')
        save_path = os.path.join(args.model_save_path, args.data_name)
        save_file_name = os.path.join(save_path, 
                                        f'checkpoint_src_{args.src_vocab_size}_trg_{args.trg_vocab_size}_v_{args.variational_mode}_p_{args.parallel}.pth.tar')
        checkpoint = torch.load(save_file_name)
        start_epoch = checkpoint['epoch'] - 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        del checkpoint

    #===================================#
    #=========Model Train Start=========#
    #===================================#

    write_log(logger, 'Traing start!')
    best_aug_val_loss = 1e+4
    best_cls_val_loss = 1e+4
    
    for epoch in range(start_epoch + 1, args.cls_num_epochs + 1):
        start_time_e = time()

        write_log(logger, 'Augmenter training start...')
        aug_model.train()

        for i, batch_iter in enumerate(tqdm(dataloader_dict['train'], bar_format='{l_bar}{bar:30}{r_bar}{bar:-2b}')):

            # Input setting
            b_iter = input_to_device(batch_iter, device=device)
            src_sequence, src_att, src_seg, trg_label = b_iter

            #===================================#
            #=======Classifier Training=========#
            #===================================#

            cls_optimizer.zero_grad(set_to_none=True)

            # encoder_out_copy = encoder_out.clone().detach().requires_grad_(True)
            # latent_out_copy = Variable(latent_out.clone(), volatile=False)

            # Encoding
            encoder_out, latent_out = aug_model.encode(src_input_ids=src_sequence, 
                                                       src_attention_mask=src_att)

            # Classify
            logit = aug_model.classify(latent_out=latent_out)
            cls_loss = cls_criterion(logit, trg_label)
            cls_loss.backward()
            if args.clip_grad_norm > 0:
                clip_grad_norm_(aug_model.parameters(), args.clip_grad_norm)
            cls_optimizer.step()
            cls_scheduler.step()

            # Print loss value only training
            if i == 0 or i % args.print_freq == 0 or i+1 == len(dataloader_dict['train']):
                cls_iter_log = "[CLS][Epoch:%03d][%03d/%03d] train_cls_loss:%03.2f | learning_rate:%1.6f | spend_time:%02.2fmin" % \
                    (epoch, i, len(dataloader_dict['train']), cls_loss.item(), cls_optimizer.param_groups[0]['lr'], (time() - start_time_e) / 60)
                write_log(logger, cls_iter_log)

        # Validation 
        aug_model.eval()

        val_cls_loss = 0
        val_acc = 0

        for batch_iter in tqdm(dataloader_dict['valid'], bar_format='{l_bar}{bar:30}{r_bar}{bar:-2b}'):

            b_iter = input_to_device(batch_iter, device=device)
            src_sequence, src_att, src_seg, trg_label = b_iter

            # Reconsturction setting
            trg_sequence_gold = src_sequence.contiguous().view(-1)
            non_pad = trg_sequence_gold != aug_model.pad_idx
        
            with torch.no_grad():
                encoder_out, latent_out = aug_model.encode(src_input_ids=src_sequence, 
                                                           src_attention_mask=src_att)
                logit = aug_model.classify(latent_out=latent_out)
                cls_loss = cls_criterion(logit, trg_label)
    
            val_cls_loss += cls_loss
            val_acc += (logit.argmax(dim=1) == trg_label.argmax(dim=1)).sum() / len(trg_label)

        val_cls_loss /= len(dataloader_dict['valid'])
        val_acc /= len(dataloader_dict['valid'])
        write_log(logger, 'Classifier Validation CrossEntropy Loss: %3.3f' % val_cls_loss)
        write_log(logger, 'Classifier Validation Accuracy: %3.2f%%' % (val_acc * 100))

        save_file_name = os.path.join(args.model_save_path, args.data_name, args.model_type, 'cls_checkpoint_only_latent.pth.tar')
        if val_cls_loss < best_cls_val_loss:
            write_log(logger, 'Classifier Checkpoint saving...')
            torch.save({
                'cls_model': cls_model.state_dict()
            }, save_file_name)
            best_cls_val_loss = val_cls_loss
            best_cls_epoch = epoch
        else:
            else_log = f'[CLS]Still {best_cls_epoch} epoch Loss({round(best_cls_val_loss.item(), 2)}) is better...'
            write_log(logger, else_log)

    for epoch in range(start_epoch + 1, args.aug_num_epochs + 1):
        start_time_e = time()

        write_log(logger, 'Augmenter training start...')
        aug_model.train()

        for i, batch_iter in enumerate(tqdm(dataloader_dict['train'], bar_format='{l_bar}{bar:30}{r_bar}{bar:-2b}')):

            # Input setting
            b_iter = input_to_device(batch_iter, device=device)
            src_sequence, src_att, src_seg, trg_label = b_iter

            #===================================#
            #=======Classifier Training=========#
            #===================================#

            aug_optimizer.zero_grad(set_to_none=True)

            # Encoding
            with torch.no_grad():
                encoder_out, latent_out = aug_model.encode(src_input_ids=src_sequence, 
                                                           src_attention_mask=src_att)

            encoder_out_copy = encoder_out.clone().detach().requires_grad_(True)
            latent_out_copy = latent_out.clone().detach().requires_grad_(True)
            decoder_out = aug_model.reconstruct(src_input_ids=src_sequence, 
                                                src_attention_mask=src_att,
                                                encoder_out=encoder_out_copy,
                                                latent_out=latent_out_copy)

            mmd_loss = torch.tensor(0.0).to(device)
            recon_loss = recon_criterion(decoder_out.view(-1, src_vocab_num), src_sequence.contiguous().view(-1))
            total_loss = mmd_loss + recon_loss + cls_loss
            total_loss.backward()
            if args.clip_grad_norm > 0:
                clip_grad_norm_(aug_model.parameters(), args.clip_grad_norm)
            aug_optimizer.step()
            aug_scheduler.step()

            # Print loss value only training
            if i == 0 or i % args.print_freq == 0 or i+1 == len(dataloader_dict['train']):
                aug_iter_log = "[AUG][Epoch:%03d][%03d/%03d] train_recon_loss:%03.2f | train_mmd_loss:%03.2f | learning_rate:%1.6f | spend_time:%02.2fmin" % \
                    (epoch, i, len(dataloader_dict['train']), recon_loss.item(), mmd_loss.item(), aug_optimizer.param_groups[0]['lr'], (time() - start_time_e) / 60)
                write_log(logger, aug_iter_log)

        # Validation 
        aug_model.eval()

        val_mmd_loss = 0
        val_recon_loss = 0

        for batch_iter in tqdm(dataloader_dict['valid'], bar_format='{l_bar}{bar:30}{r_bar}{bar:-2b}'):

            # Input setting
            b_iter = input_to_device(batch_iter, device=device)
            src_sequence, src_att, src_seg, trg_label = b_iter

            # Encoding
            with torch.no_grad():
                encoder_out, latent_out = aug_model.encode(src_input_ids=src_sequence, 
                                                           src_attention_mask=src_att)
                decoder_out = aug_model.reconstruct(src_input_ids=src_sequence, 
                                                    src_attention_mask=src_att,
                                                    encoder_out=encoder_out,
                                                    latent_out=latent_out)

            # mmd_loss = compute_mmd(latent_out, z_var=args.z_variation) * 100
            mmd_loss = torch.tensor(0.0).to(device)
            recon_loss = recon_criterion(decoder_out.view(-1, src_vocab_num), src_sequence.contiguous().view(-1))
            val_mmd_loss += mmd_loss
            val_recon_loss += recon_loss

        val_recon_loss /= len(dataloader_dict['valid'])
        val_mmd_loss /= len(dataloader_dict['valid'])
        write_log(logger, 'Augmenter Validation CrossEntropy Loss: %3.3f' % val_recon_loss)
        write_log(logger, 'Augmenter Validation MMD Loss: %3.3f' % val_mmd_loss)

        trg_label = torch.flip(trg_label, dims=[1]).to(device)
        src_output = aug_model.tokenizer.batch_decode(src_sequence, skip_special_tokens=True)[0]
        origin_output = aug_model.tokenizer.batch_decode(decoder_out.argmax(dim=2), skip_special_tokens=True)[0]
        
        eps_dict = dict()
        for epsilon in [2, 5, 8]: # * 0.9
            with torch.no_grad():
                encoder_out, latent_out = aug_model.encode(src_input_ids=src_sequence, 
                                                        src_attention_mask=src_att)

            encoder_out_copy = Variable(encoder_out.clone(), volatile=False)
            data = Variable(latent_out.clone(), volatile=False)
            data.requires_grad = True

            logit = aug_model.classify(latent_out=data)

            cls_loss = cls_criterion(logit, trg_label)
            aug_model.zero_grad()
            cls_loss.backward()
            data_grad = data.grad
            data = data - ((epsilon * 0.8) * data_grad)

            with torch.no_grad():
                decoder_out = aug_model.generate(src_input_ids=src_sequence, 
                                                 src_attention_mask=src_att,
                                                 encoder_out=encoder_out,
                                                 latent_out=data)

            # Augmenting
            eps_dict[f'eps_{epsilon}'] = aug_model.tokenizer.batch_decode(decoder_out, skip_special_tokens=True)[0]

        write_log(logger, 'Generated Examples')
        write_log(logger, f'Source: {src_output}')
        write_log(logger, f'Augmented_origin: {origin_output}')
        write_log(logger, f'Augmented_2: {eps_dict["eps_2"]}')
        write_log(logger, f'Augmented_5: {eps_dict["eps_5"]}')
        write_log(logger, f'Augmented_8: {eps_dict["eps_8"]}')

        save_file_name = os.path.join(args.model_save_path, args.data_name, args.model_type, 'aug_checkpoint_only_latent.pth.tar')
        if val_recon_loss < best_aug_val_loss:
            write_log(logger, 'Augmenter Checkpoint saving...')
            torch.save({
                'aug_model': aug_model.state_dict()
            }, save_file_name)
            best_aug_val_loss = val_recon_loss
            best_aug_epoch = epoch
        else:
            else_log = f'[AUG]Still {best_aug_epoch} epoch Loss({round(best_aug_val_loss.item(), 2)}) is better...'
            write_log(logger, else_log)

    # 3) Results
    write_log(logger, f'Best AUG Epoch: {best_aug_epoch}')
    write_log(logger, f'Best AUG Loss: {round(best_aug_val_loss.item(), 2)}')
    write_log(logger, f'Best CLS Epoch: {best_cls_epoch}')
    write_log(logger, f'Best CLS Loss: {round(best_cls_val_loss.item(), 2)}')