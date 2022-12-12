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
from transformers import AutoTokenizer
# Import custom modules
from model.model import TransformerModel
from model.dataset import CustomDataset
from model.loss import compute_mmd
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

    save_path = os.path.join(args.preprocess_path, args.data_name)

    with h5py.File(os.path.join(save_path, args.cls_model, 'processed.hdf5'), 'r') as f:
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

    with open(os.path.join(save_path, args.cls_model, 'word2id.pkl'), 'rb') as f:
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
    model = TransformerModel(model_type='bert', num_labels=num_labels,
                             isPreTrain=args.isPreTrain, z_variation=args.z_variation,
                          dropout=args.dropout)
    model.to(device)

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
    optimizer = optimizer_select(optimizer_model=args.optimizer, model=model, args=args)
    scheduler = shceduler_select(optimizer=optimizer, dataloader_dict=dataloader_dict, args=args)
    # total_iters = round(len(dataloader_dict['aug_train'])/args.num_grad_accumulate*args.num_epochs)
    # cls_scheduler = get_cosine_schedule_with_warmup(cls_optimizer, round(cls_total_iters*0.3), cls_total_iters) # args.warmup_ratio = 0.3 -> Need to fix

    cudnn.benchmark = True
    scaler = GradScaler()
    softmax = nn.Softmax(dim=1)
    cls_metric = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing_eps).to(device)
    recon_metric = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing_eps, ignore_index=model.pad_idx).to(device)

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
    cls_train_iter = iter(dataloader_dict['train'])
    aug_train_iter = iter(dataloader_dict['aug_train'])
    
    for epoch in range(start_epoch + 1, args.num_epochs + 1):
        start_time_e = time()
        finish_epoch = False
        freq = 0

        # Training step
        model.train()

        while True:

            #===================================#
            #========Classifier Training========#
            #===================================#
            optimizer.zero_grad(set_to_none=True)

            for _ in range(args.num_grad_accumulate):

                try:
                    batch_iter = next(cls_train_iter)
                except StopIteration:
                    train_iter = iter(dataloader_dict['train'])
                    batch_iter = next(train_iter)

                # Input setting
                b_iter = input_to_device(batch_iter, device=device)
                src_sequence, src_att, src_seg, trg_label = b_iter

                # Classifier training
                with autocast():
                    decoder_out, logit = model(src_input_ids=src_sequence,
                                               src_attention_mask=src_att,
                                               src_token_type_ids=src_seg)
                    cls_loss = cls_metric(logit, trg_label)/args.num_grad_accumulate

                scaler.scale(cls_loss).backward()
                
            scaler.step(optimizer)
            scaler.update()

            #===================================#
            #========Augmenter Training=========#
            #===================================#
            optimizer.zero_grad(set_to_none=True)

            for _ in range(args.num_grad_accumulate):

                try:
                    aug_batch_iter = next(aug_train_iter)
                except StopIteration:
                    aug_train_iter = iter(dataloader_dict['aug_train'])
                    aug_batch_iter = next(aug_train_iter)
                    finish_epoch = True

                # Input setting
                aug_b_iter = input_to_device(aug_batch_iter, device=device)
                aug_src_sequence, aug_src_att, aug_src_seg, aug_trg_label = aug_b_iter

                # Augmenter training
                with autocast():
                    encoder_out, decoder_out, z = model.augment(src_input_ids=aug_src_sequence, 
                                                                src_attention_mask=aug_src_att,
                                                                src_token_type_ids=aug_src_seg)
                    mmd_loss = compute_mmd(z.view(args.batch_size, -1), 
                                           z_var=args.z_variation) * 10
                    recon_loss = recon_metric(decoder_out.view(-1, src_vocab_num), aug_src_sequence.contiguous().view(-1))

                # Augmenting
                # decoder_out_token = decoder_out.argmax(dim=2)
                # augmented_output = model.tokenizer.batch_decode(decoder_out_token, skip_special_tokens=True)
                # augmented_tokenized = model.tokenizer(augmented_output, return_tensors='pt', 
                #                                       max_length=args.src_max_len, padding='max_length', truncation=True)
                # ood_trg_list = torch.full((len(augmented_output), num_labels), 1 / num_labels).to(device)

                # with autocast():
                #     logit = model.classify_(src_input_ids=augmented_tokenized['input_ids'].to(device),
                #                             src_attention_mask=augmented_tokenized['attention_mask'].to(device),
                #                             src_token_type_ids=augmented_tokenized['token_type_ids'].to(device))
                #     confidence = softmax(logit)
                decoder_out_token = decoder_out.argmax(dim=2)
                ood_trg_list = torch.full((len(decoder_out_token), num_labels), 1 / num_labels).to(device)

                model.eval()

                with autocast():
                    logit = model.classify_(src_input_ids=decoder_out_token,
                                            src_attention_mask=aug_src_att,
                                            src_token_type_ids=aug_src_seg)
                    confidence = softmax(logit)

                model.train()

                new_loss = F.cross_entropy(confidence, ood_trg_list)
        
                total_loss = mmd_loss + recon_loss + new_loss
                scaler.scale(total_loss).backward()

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
                
            # Print loss value only training
            if freq * args.num_grad_accumulate % args.print_freq == 0 or finish_epoch:
                printing_freq = freq * args.num_grad_accumulate
                iter_log = "[Epoch:%03d][%03d/%03d] train_cls_loss:%03.2f | train_recon_loss:%03.2f | train_mmd_loss:%03.2f | train_new_loss:%03.2f | learning_rate:%1.6f | spend_time:%02.2fmin" % \
                    (epoch, printing_freq, len(dataloader_dict['train']), cls_loss.item(), recon_loss.item(), mmd_loss.item(), new_loss.item(), optimizer.param_groups[0]['lr'], (time() - start_time_e) / 60)
                write_log(logger, iter_log)
            freq += 1


            if finish_epoch:
                break

        # Validation 
        model.eval()

        val_mmd_loss = 0
        val_recon_loss = 0
        val_new_loss = 0
        val_cls_loss = 0
        val_acc = 0

        #===================================#
        #=======Augmenter Validation========#
        #===================================#
        write_log(logger, 'Augmenter Validation start...')
        for aug_batch_iter in tqdm(dataloader_dict['valid'], bar_format='{l_bar}{bar:30}{r_bar}{bar:-2b}'):

            aug_b_iter = input_to_device(aug_batch_iter, device=device)
            aug_src_sequence, aug_src_att, aug_src_seg, _ = aug_b_iter

            # Reconsturction setting
            trg_sequence_gold = aug_src_sequence.contiguous().view(-1)
            non_pad = trg_sequence_gold != model.pad_idx
        
            with autocast():
                decoder_out, z = model.generate(src_input_ids=aug_src_sequence, 
                                                src_attention_mask=aug_src_att,
                                                src_token_type_ids=aug_src_seg)
                mmd_loss = compute_mmd(z.view(args.batch_size, -1), 
                                       z_var=args.z_variation) * 10
                recon_loss = F.cross_entropy(decoder_out.view(-1, src_vocab_num), trg_sequence_gold)
                val_mmd_loss += mmd_loss
                val_recon_loss += recon_loss
                val_acc += (decoder_out.argmax(dim=2).view(-1)[non_pad] == trg_sequence_gold[non_pad]).sum() / len(trg_sequence_gold[non_pad])

            decoder_out_token = decoder_out.argmax(dim=2)
            augmented_output = model.tokenizer.batch_decode(decoder_out_token, skip_special_tokens=True)
            augmented_tokenized = model.tokenizer(augmented_output, return_tensors='pt', 
                                                    max_length=args.src_max_len, padding='max_length', truncation=True)
            ood_trg_list = torch.full((len(augmented_output), num_labels), 1 / num_labels).to(device)

            with torch.no_grad():
                with autocast():
                    logit = model.classify_(src_input_ids=decoder_out_token,
                                            src_attention_mask=aug_src_att,
                                            src_token_type_ids=aug_src_seg)
                    confidence = softmax(logit)

            val_new_loss += F.cross_entropy(confidence, ood_trg_list)

        val_recon_loss /= len(dataloader_dict['valid'])
        val_mmd_loss /= len(dataloader_dict['valid'])
        val_new_loss /= len(dataloader_dict['valid'])
        val_acc /= len(dataloader_dict['valid'])
        write_log(logger, 'Augmenter Validation CrossEntropy Loss: %3.3f' % val_recon_loss)
        write_log(logger, 'Augmenter Validation MMD Loss: %3.3f' % val_mmd_loss)
        write_log(logger, 'Augmenter Validation Reconstruction Accuracy: %3.2f%%' % (val_acc * 100))

        #===================================#
        #=======Classifier Validation=======#
        #===================================#
        write_log(logger, 'Classifier validation start...')
        for batch_iter in tqdm(dataloader_dict['valid'], bar_format='{l_bar}{bar:30}{r_bar}{bar:-2b}'):

            b_iter = input_to_device(batch_iter, device=device)
            src_sequence, src_att, src_seg, trg_label = b_iter

            with torch.no_grad():
                with autocast():
                    logit = model.classify_(src_input_ids=src_sequence,
                                            src_attention_mask=src_att,
                                            src_token_type_ids=src_seg)
                    cls_loss = cls_metric(logit, trg_label)
            
            val_cls_loss += cls_loss
            val_acc += (logit.argmax(dim=1) == trg_label.argmax(dim=1)).sum() / len(trg_label)

        val_cls_loss /= len(dataloader_dict['valid'])
        val_acc /= len(dataloader_dict['valid'])
        write_log(logger, 'Classifier Validation CrossEntropy Loss: %3.3f' % val_cls_loss)
        write_log(logger, 'Classifier Validation Accuracy: %3.2f%%' % (val_acc * 100))

        #===================================#
        #=========Text Augmentation=========#
        #===================================#
        for i, batch_iter in enumerate(tqdm(dataloader_dict['aug_train'], bar_format='{l_bar}{bar:30}{r_bar}{bar:-2b}')):
            b_iter = input_to_device(batch_iter, device=device)
            aug_src_sequence, aug_src_att, aug_src_seg, trg_label = b_iter
            trg_label = trg_label.cpu().numpy()

            # Reconsturction setting
            trg_sequence_gold = aug_src_sequence.contiguous().view(-1)
            non_pad = trg_sequence_gold != model.pad_idx
        
            with torch.no_grad():
                with autocast():
                    decoder_out, z = model.generate(src_input_ids=aug_src_sequence, 
                                                    src_attention_mask=aug_src_att,
                                                    src_token_type_ids=aug_src_seg)

            # Augmenting
            decoder_out_token = decoder_out.argmax(dim=2)
            augmented_output = model.tokenizer.batch_decode(decoder_out_token, skip_special_tokens=True)
            augmented_tokenized = model.tokenizer(augmented_output, return_tensors='np',
                                                  max_length=args.src_max_len, padding='max_length', truncation=True)
            train_src_input_ids = np.concatenate((train_src_input_ids, augmented_tokenized['input_ids']))
            train_src_attention_mask = np.concatenate((train_src_attention_mask, augmented_tokenized['attention_mask']))
            train_src_token_type_ids = np.concatenate((train_src_token_type_ids, augmented_tokenized['token_type_ids']))
            train_trg_list = np.concatenate((train_trg_list, trg_label))

            if i % int(len(dataloader_dict['aug_train'])*0.3) == 0:
                src_token = model.tokenizer.batch_decode(aug_src_sequence, skip_special_tokens=True)[0]
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
        if val_mmd_loss < best_val_loss:
            write_log(logger, 'Checkpoint saving...')
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict()
            }, save_file_name)
            best_val_loss = val_mmd_loss
            best_epoch = epoch
        else:
            else_log = f'Still {best_epoch} epoch Loss({round(best_val_loss.item(), 2)}) is better...'
            write_log(logger, else_log)

    # 3) Results
    write_log(logger, f'Best Epoch: {best_epoch}')
    write_log(logger, f'Best Loss: {round(best_val_loss.item(), 2)}')