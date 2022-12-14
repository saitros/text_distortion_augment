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
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# Import custom modules
from model.model import TransformerModel
from model.dataset import CustomDataset
from model.loss import compute_mmd, CustomLoss
from optimizer.utils import shceduler_select, optimizer_select
from optimizer.scheduler import get_cosine_schedule_with_warmup
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

    save_path = os.path.join(args.preprocess_path, args.data_name, args.model_type)

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
    cls_model = AutoModelForSequenceClassification.from_pretrained('bert-base-cased', num_labels=num_labels)
    aug_model = TransformerModel(model_type='bert', num_labels=num_labels,
                                 isPreTrain=args.isPreTrain, z_variation=args.z_variation,
                                 dropout=args.dropout)
    cls_model.to(device)
    aug_model.to(device)

    # 2) Dataloader setting
    dataset_dict = {
        'train': CustomDataset(src_list=train_src_input_ids, src_att_list=train_src_attention_mask,
                               src_seg_list=train_src_token_type_ids,
                               trg_list=train_trg_list, src_max_len=args.src_max_len),
        'aug_train': CustomDataset(src_list=train_src_input_ids, src_att_list=train_src_attention_mask,
                                   src_seg_list=train_src_token_type_ids,
                                   trg_list=train_trg_list, src_max_len=args.src_max_len),
        'valid': CustomDataset(src_list=valid_src_input_ids, src_att_list=valid_src_attention_mask,
                               src_seg_list=valid_src_token_type_ids,
                               trg_list=valid_trg_list, src_max_len=args.src_max_len)
    }
    dataloader_dict = {
        'train': DataLoader(dataset_dict['train'], drop_last=False,
                            batch_size=args.batch_size, shuffle=True, pin_memory=True,
                            num_workers=args.num_workers),
        'aug_train': DataLoader(dataset_dict['aug_train'], drop_last=False,
                                batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                num_workers=args.num_workers),
        'valid': DataLoader(dataset_dict['valid'], drop_last=False,
                            batch_size=args.batch_size, shuffle=True, pin_memory=True,
                            num_workers=args.num_workers)
    }
    write_log(logger, f"Total number of trainingsets  iterations - {len(dataset_dict['train'])}, {len(dataloader_dict['train'])}")

    # del (
    #     train_src_input_ids, train_src_attention_mask, train_src_token_type_ids, train_trg_list,
    #     valid_src_input_ids, valid_src_attention_mask, valid_src_token_type_ids, valid_trg_list
    # )

    # 3) Optimizer & Learning rate scheduler setting
    cls_optimizer = optimizer_select(optimizer_model=args.cls_optimizer, model=cls_model, lr=args.cls_lr)
    aug_optimizer = optimizer_select(optimizer_model=args.aug_optimizer, model=aug_model, lr=args.aug_lr)
    cls_scheduler = shceduler_select(scheduler_model=args.cls_scheduler, optimizer=cls_optimizer, dataloader_len=len(dataloader_dict['train']), args=args)
    aug_scheduler = shceduler_select(scheduler_model='constant', optimizer=aug_optimizer, dataloader_len=len(dataloader_dict['train']), args=args)
    cls_scaler = GradScaler()
    aug_scaler = GradScaler()
    # total_iters = round(len(dataloader_dict['train'])/args.num_grad_accumulate*args.num_epochs)
    # scheduler = get_cosine_schedule_with_warmup(optimizer, round(total_iters*0.3), total_iters) # args.warmup_ratio = 0.3 -> Need to fix

    cudnn.benchmark = True
    softmax = nn.Softmax(dim=1)
    cls_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing_eps).to(device)
    recon_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing_eps, ignore_index=aug_model.pad_idx).to(device)

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
        scaler.load_state_dict(checkpoint['scaler'])
        del checkpoint

    #===================================#
    #=========Model Train Start=========#
    #===================================#

    write_log(logger, 'Traing start!')
    best_val_loss = 1e+3
    
    for epoch in range(start_epoch + 1, args.num_epochs + 1):
        start_time_e = time()

        # Training step
        cls_model.train()

        write_log(logger, 'Classifier training start...')
        for i, batch_iter in enumerate(tqdm(dataloader_dict['train'], bar_format='{l_bar}{bar:30}{r_bar}{bar:-2b}')):

            cls_optimizer.zero_grad(set_to_none=True)

            # Input setting
            b_iter = input_to_device(batch_iter, device=device)
            src_sequence, src_att, src_seg, trg_label = b_iter

            # Classifier training
            with autocast():
                logit = cls_model(input_ids=src_sequence,
                                  attention_mask=src_att,
                                  token_type_ids=src_seg)
                logit = logit['logits']
                cls_loss = cls_criterion(logit, trg_label)

            cls_scaler.scale(cls_loss).backward()
            cls_scaler.step(cls_optimizer)
            cls_scaler.update()
            cls_scheduler.step()

            # Print loss value only training
            if i == 0 or i % args.print_freq == 0 or i == len(dataloader_dict['train'])-1:
                iter_log = "[CLS][Epoch:%03d][%03d/%03d] train_cls_loss:%03.2f | learning_rate:%1.6f | spend_time:%02.2fmin" % \
                    (epoch, i, len(dataloader_dict['train'])-1, cls_loss.item(), cls_optimizer.param_groups[0]['lr'], (time() - start_time_e) / 60)
                write_log(logger, iter_log)

        # Validation 
        cls_model.eval()
        val_cls_loss = 0
        val_acc = 0

        #===================================#
        #=======Classifier Validation=======#
        #===================================#
        write_log(logger, 'Classifier validation start...')
        for batch_iter in tqdm(dataloader_dict['valid'], bar_format='{l_bar}{bar:30}{r_bar}{bar:-2b}'):

            b_iter = input_to_device(batch_iter, device=device)
            src_sequence, src_att, src_seg, trg_label = b_iter

            with torch.no_grad():
                with autocast():
                    logit = cls_model(input_ids=src_sequence,
                                  attention_mask=src_att,
                                  token_type_ids=src_seg)
                    logit = logit['logits']
                    cls_loss = cls_criterion(logit, trg_label)
            
            val_cls_loss += cls_loss
            val_acc += (logit.argmax(dim=1) == trg_label.argmax(dim=1)).sum() / len(trg_label)

        val_cls_loss /= len(dataloader_dict['valid'])
        val_acc /= len(dataloader_dict['valid'])
        write_log(logger, 'Classifier Validation CrossEntropy Loss: %3.3f' % val_cls_loss)
        write_log(logger, 'Classifier Validation Accuracy: %3.2f%%' % (val_acc * 100))

        write_log(logger, 'Augmenter training start...')
        aug_model.train()
        CustomCriterion = CustomLoss(model=cls_model, device=device, num_labels=num_labels).to(device)

        for i, batch_iter in enumerate(tqdm(dataloader_dict['aug_train'], bar_format='{l_bar}{bar:30}{r_bar}{bar:-2b}')):

            aug_optimizer.zero_grad(set_to_none=True)

            # Input setting
            b_iter = input_to_device(batch_iter, device=device)
            src_sequence, src_att, src_seg, trg_label = b_iter

            # Augmenter training
            with autocast():
                decoder_out, z = aug_model(src_input_ids=src_sequence, 
                                                src_attention_mask=src_att,
                                                src_token_type_ids=src_seg)
                mmd_loss = compute_mmd(z.view(args.batch_size, -1), 
                                        z_var=args.z_variation) * 100
                recon_loss = recon_criterion(decoder_out.view(-1, src_vocab_num), src_sequence.contiguous().view(-1))

            decoder_out_token = decoder_out.argmax(dim=2)

            new_loss = CustomCriterion(src_sequence=decoder_out_token, src_att=src_att, src_seg=src_seg)

            total_loss = mmd_loss + recon_loss + new_loss
            aug_scaler.scale(total_loss).backward()
            aug_scaler.step(aug_optimizer)
            aug_scaler.update()
            aug_scheduler.step()
                
            # Print loss value only training
            if i == 0 or i % args.print_freq == 0 or i == len(dataloader_dict['train'])-1:
                iter_log = "[AUG][Epoch:%03d][%03d/%03d] train_recon_loss:%03.2f | train_mmd_loss:%03.2f | train_new_loss:%03.2f | learning_rate:%1.6f | spend_time:%02.2fmin" % \
                    (epoch, i, len(dataloader_dict['train'])-1, recon_loss.item(), mmd_loss.item(), new_loss.item(), aug_optimizer.param_groups[0]['lr'], (time() - start_time_e) / 60)
                write_log(logger, iter_log)

        # Validation 
        aug_model.eval()
        val_mmd_loss = 0
        val_recon_loss = 0
        val_new_loss = 0

        #===================================#
        #=======Augmenter Validation========#
        #===================================#
        write_log(logger, 'Augmenter Validation start...')
        for batch_iter in tqdm(dataloader_dict['valid'], bar_format='{l_bar}{bar:30}{r_bar}{bar:-2b}'):

            b_iter = input_to_device(batch_iter, device=device)
            src_sequence, src_att, src_seg, _ = b_iter

            # Reconsturction setting
            trg_sequence_gold = src_sequence.contiguous().view(-1)
            non_pad = trg_sequence_gold != aug_model.pad_idx
        
            with torch.no_grad():
                # Augmenter training
                with autocast():
                    decoder_out, z = aug_model(src_input_ids=src_sequence, 
                                                   src_attention_mask=src_att,
                                                   src_token_type_ids=src_seg)
                    mmd_loss = compute_mmd(z.view(args.batch_size, -1), 
                                           z_var=args.z_variation) * 100
                    recon_loss = recon_criterion(decoder_out.view(-1, src_vocab_num), src_sequence.contiguous().view(-1))

                decoder_out_token = decoder_out.argmax(dim=2)

                new_loss = CustomCriterion(src_sequence=decoder_out_token, src_att=src_att, src_seg=src_seg)

                val_mmd_loss += mmd_loss
                val_recon_loss += recon_loss
                val_new_loss += new_loss

        val_recon_loss /= len(dataloader_dict['valid'])
        val_mmd_loss /= len(dataloader_dict['valid'])
        val_new_loss /= len(dataloader_dict['valid'])
        write_log(logger, 'Augmenter Validation CrossEntropy Loss: %3.3f' % val_recon_loss)
        write_log(logger, 'Augmenter Validation MMD Loss: %3.3f' % val_mmd_loss)
        write_log(logger, 'Augmenter Validation Custom Loss: %3.3f' % val_new_loss)

        #===================================#
        #=========Text Augmentation=========#
        #===================================#
        for i, batch_iter in enumerate(tqdm(dataloader_dict['aug_train'], bar_format='{l_bar}{bar:30}{r_bar}{bar:-2b}')):

            b_iter = input_to_device(batch_iter, device=device)
            aug_src_sequence, aug_src_att, aug_src_seg, trg_label = b_iter
            trg_label = trg_label.cpu().numpy()

            with torch.no_grad():
                with autocast():
                    decoder_out, z = aug_model(src_input_ids=aug_src_sequence, 
                                                   src_attention_mask=aug_src_att,
                                                   src_token_type_ids=aug_src_seg)

            # Augmenting
            decoder_out_token = decoder_out.argmax(dim=2)
            augmented_output = aug_model.tokenizer.batch_decode(decoder_out_token, skip_special_tokens=True)
            augmented_tokenized = aug_model.tokenizer(augmented_output, return_tensors='np',
                                                  max_length=args.src_max_len, padding='max_length', truncation=True)
            train_src_input_ids = np.concatenate((train_src_input_ids, augmented_tokenized['input_ids']))
            train_src_attention_mask = np.concatenate((train_src_attention_mask, augmented_tokenized['attention_mask']))
            train_src_token_type_ids = np.concatenate((train_src_token_type_ids, augmented_tokenized['token_type_ids']))
            train_trg_list = np.concatenate((train_trg_list, trg_label))

            if i % int(len(dataloader_dict['aug_train'])*0.3) == 0:
                src_token = aug_model.tokenizer.batch_decode(aug_src_sequence, skip_special_tokens=True)[0]
                aug_token = augmented_output[0]
                write_log(logger, 'Generated Examples')
                write_log(logger, f'Source: {src_token}')
                write_log(logger, f'Augmented: {aug_token}')

        dataset_dict['train'] = CustomDataset(src_list=train_src_input_ids, src_att_list=train_src_attention_mask,
                                              src_seg_list=train_src_token_type_ids,
                                              trg_list=train_trg_list, src_max_len=args.src_max_len)
        dataloader_dict['train'] = DataLoader(dataset_dict['train'], drop_last=False,
                                              batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                              num_workers=args.num_workers)

        save_file_name = os.path.join(args.model_save_path, args.data_name)
        save_file_name += 'checkpoint.pth.tar'
        if val_cls_loss < best_val_loss:
            write_log(logger, 'Checkpoint saving...')
            torch.save({
                'epoch': epoch,
                'model': cls_model.state_dict(),
                'optimizer': cls_optimizer.state_dict(),
                'scheduler': cls_scheduler.state_dict(),
                'scaler': cls_scaler.state_dict()
            }, save_file_name)
            best_val_loss = val_cls_loss
            best_epoch = epoch
        else:
            else_log = f'Still {best_epoch} epoch Loss({round(best_val_loss.item(), 2)}) is better...'
            write_log(logger, else_log)

    # 3) Results
    write_log(logger, f'Best Epoch: {best_epoch}')
    write_log(logger, f'Best Loss: {round(best_val_loss.item(), 2)}')