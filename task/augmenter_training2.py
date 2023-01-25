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
    model = TransformerModel(model_type=args.model_type, src_max_len=args.src_max_len,
                                 isPreTrain=args.isPreTrain, num_labels=num_labels, dropout=args.dropout)
    cls_model = ClassifierModel(d_latent=model.d_hidden, num_labels=num_labels, dropout=args.dropout)
    model.to(device)
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
        'train': DataLoader(dataset_dict['train'], drop_last=True,
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
    cls_optimizer = optimizer_select(optimizer_model=args.cls_optimizer, model=cls_model, lr=args.cls_lr, w_decay=args.w_decay)
    aug_optimizer = optimizer_select(optimizer_model=args.aug_optimizer, model=model, lr=args.aug_lr, w_decay=args.w_decay)
    cls_scheduler = shceduler_select(phase='cls', scheduler_model=args.cls_scheduler, optimizer=cls_optimizer, dataloader_len=len(dataloader_dict['train']), args=args)
    aug_scheduler = shceduler_select(phase='aug', scheduler_model=args.aug_scheduler, optimizer=aug_optimizer, dataloader_len=len(dataloader_dict['train']), args=args)

    cudnn.benchmark = True
    cls_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing_eps).to(device)
    recon_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing_eps, ignore_index=model.pad_idx).to(device)

    # 3) Model resume
    start_epoch = 0
    cls_training_done = False
    if args.resume:
        write_log(logger, 'Resume model...')
        save_file_name = os.path.join(args.model_save_path, args.data_name, args.model_type, 'checkpoint.pth.tar')
        checkpoint = torch.load(save_file_name)
        cls_training_done = checkpoint['cls_training_done']
        start_epoch = checkpoint['epoch'] - 1
        model.load_state_dict(checkpoint['model'])
        cls_optimizer.load_state_dict(checkpoint['cls_optimizer'])
        aug_optimizer.load_state_dict(checkpoint['aug_optimizer'])
        cls_scheduler.load_state_dict(checkpoint['cls_scheduler'])
        aug_scheduler.load_state_dict(checkpoint['aug_scheduler'])
        del checkpoint

    #===================================#
    #=========Model Train Start=========#
    #===================================#

    write_log(logger, 'Traing start!')
    best_aug_val_loss = 1e+4
    best_cls_val_loss = 1e+4
    
    for epoch in range(start_epoch + 1, args.cls_num_epochs + 1):
        start_time_e = time()

        if cls_training_done:
            break

        write_log(logger, 'Classifier training start...')
        model.train()
        cls_model.train()

        for i, batch_iter in enumerate(tqdm(dataloader_dict['train'], bar_format='{l_bar}{bar:30}{r_bar}{bar:-2b}')):

            # Input setting
            b_iter = input_to_device(batch_iter, device=device)
            src_sequence, src_att, src_seg, trg_label = b_iter

            # Encoding
            encoder_out = model.encode(input_ids=src_sequence, attention_mask=src_att)
            latent_out, latent_encoder_out = model.latent_encode(encoder_out=encoder_out)

            # Reconstruction
            recon_out = model(input_ids=src_sequence, attention_mask=src_att, encoder_out=encoder_out, latent_out=latent_out)
            recon_loss = recon_criterion(recon_out.view(-1, src_vocab_num), src_sequence.contiguous().view(-1))
            # mmd_loss = compute_mmd(latent_encoder_out, z_var=args.z_variation) * 100
            mmd_loss = torch.tensor(0)

            # Loss Backward
            aug_optimizer.zero_grad(set_to_none=True)
            total_recon_loss = recon_loss + mmd_loss
            recon_loss.backward()
            if args.clip_grad_norm > 0:
                clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            aug_optimizer.step()
            aug_scheduler.step()

            # Classifier
            classifier_out = cls_model(hidden_state=Variable(latent_out.clone(), volatile=False))
            cls_loss = cls_criterion(classifier_out, trg_label)
            
            # Loss Backward
            cls_optimizer.zero_grad(set_to_none=True)
            cls_loss.backward()
            if args.clip_grad_norm > 0:
                clip_grad_norm_(cls_model.parameters(), args.clip_grad_norm)
            cls_optimizer.step()
            cls_scheduler.step()

            # Print loss value only training
            if i == 0 or i % args.print_freq == 0 or i == len(dataloader_dict['train'])-1:
                train_acc = (classifier_out.argmax(dim=1) == trg_label.argmax(dim=1)).sum() / len(trg_label)
                iter_log = "[Epoch:%03d][%03d/%03d] train_recon_loss:%03.2f | train_cls_loss:%03.2f | train_accuracy:%03.2f | train_mmd_loss:%03.2f | learning_rate:%1.6f |spend_time:%02.2fmin" % \
                    (epoch, i, len(dataloader_dict['train'])-1, recon_loss.item(), cls_loss.item(), train_acc.item(), mmd_loss.item(), cls_optimizer.param_groups[0]['lr'], (time() - start_time_e) / 60)
                write_log(logger, iter_log)

            if args.debuging_mode:
                break

        write_log(logger, 'Classifier validation start...')
        model.eval()
        val_cls_loss = 0
        val_acc = 0
        val_mmd_loss = 0
        val_recon_loss = 0

        for batch_iter in tqdm(dataloader_dict['valid'], bar_format='{l_bar}{bar:30}{r_bar}{bar:-2b}'):

            b_iter = input_to_device(batch_iter, device=device)
            src_sequence, src_att, src_seg, trg_label = b_iter
        
            with torch.no_grad():
                encoder_out = model.encode(input_ids=src_sequence, attention_mask=src_att)
                latent_out, latent_encoder_out = model.latent_encode(encoder_out=encoder_out)

                recon_out = model(input_ids=src_sequence, attention_mask=src_att, encoder_out=encoder_out, latent_out=latent_out)
                recon_loss = recon_criterion(recon_out.view(-1, src_vocab_num), src_sequence.contiguous().view(-1))
                mmd_loss = compute_mmd(latent_encoder_out, z_var=args.z_variation) * 100

                # Classifier
                classifier_out = cls_model(hidden_state=latent_out)
                cls_loss = cls_criterion(classifier_out, trg_label)

            # Loss and Accuracy Check
            val_acc += (classifier_out.argmax(dim=1) == trg_label.argmax(dim=1)).sum() / len(trg_label)
            val_cls_loss += cls_loss
            val_mmd_loss += mmd_loss
            val_recon_loss += recon_loss

            if args.debuging_mode:
                break

        val_mmd_loss /= len(dataloader_dict['valid'])
        val_cls_loss /= len(dataloader_dict['valid'])
        val_acc /= len(dataloader_dict['valid'])
        val_recon_loss /= len(dataloader_dict['valid'])
        write_log(logger, 'Augmenter Validation CrossEntropy Loss: %3.3f' % val_recon_loss)
        write_log(logger, 'Classifier Validation MMD Loss: %3.3f' % val_mmd_loss)
        write_log(logger, 'Classifier Validation CrossEntropy Loss: %3.3f' % val_cls_loss)
        write_log(logger, 'Classifier Validation Accuracy: %3.2f%%' % (val_acc * 100))

        save_file_name = os.path.join(args.model_save_path, args.data_name, args.model_type, 'checkpoint2.pth.tar')
        if val_cls_loss < best_cls_val_loss:
            write_log(logger, 'Augmenter Checkpoint saving...')
            torch.save({
                'cls_training_done': False,
                'epoch': epoch,
                'model': model.state_dict(),
                'cls_optimizer': cls_optimizer.state_dict(),
                'aug_optimizer': aug_optimizer.state_dict(),
                'cls_scheduler': cls_scheduler.state_dict(),
                'aug_scheduler': aug_scheduler.state_dict(),
            }, save_file_name)
            best_cls_val_loss = val_cls_loss
            best_cls_epoch = epoch
        else:
            else_log = f'Still {best_cls_epoch} epoch Loss({round(best_cls_val_loss.item(), 2)}) is better...'
            write_log(logger, else_log)

        #===================================#
        #=========Text Augmentation=========#
        #===================================#

        eps_dict, prob_dict = dict(), dict()

        for phase in ['train', 'valid']:
            example_iter = next(iter(dataloader_dict[phase]))

            example_iter_ = input_to_device(example_iter, device=device)
            src_sequence, src_att, src_seg, trg_label = example_iter_

            src_output = model.tokenizer.batch_decode(src_sequence, skip_special_tokens=True)[0]

            # Target Label Setting
            trg_label = torch.flip(trg_label, dims=[1]).to(device)

            # Original Reconstruction
            # with torch.no_grad():
            encoder_out = model.encode(input_ids=src_sequence, attention_mask=src_att)
            latent_out, _ = model.latent_encode(encoder_out=encoder_out)
            recon_out = model(input_ids=src_sequence, attention_mask=src_att, encoder_out=encoder_out, latent_out=latent_out)

            # Reconstruction Output Tokenizing & Pre-processing
            eps_dict['eps_0'] = model.tokenizer.batch_decode(recon_out.argmax(dim=2), skip_special_tokens=True)[0]
            inp_dict = model.tokenizer(eps_dict['eps_0'], max_length=args.src_max_len, 
                                       padding='max_length', truncation=True, return_tensors='pt')
            # Probability Calculate
            with torch.no_grad():
                encoder_out_eps_0 = model.encode(input_ids=inp_dict['input_ids'].to(device), 
                                                 attention_mask=inp_dict['attention_mask'].to(device))
                latent_out_eps_0, _ = model.latent_encode(encoder_out=encoder_out_eps_0)
                classifier_out = cls_model(hidden_state=latent_out_eps_0)
            prob_dict['eps_0'] = F.softmax(classifier_out)

            # Iterative Latent Encoding
            # encoder_out_copy = encoder_out.clone().detach().requires_grad_(True)
            # latent_out_copy, _ = model.latent_encode(encoder_out=encoder_out_copy)
            # latent_out_copy = Variable(latent_out.clone(), volatile=False)

            latent_out_copy = Variable(latent_out.clone(), volatile=False)
            latent_out_copy.requires_grad = True
            classifier_out = cls_model(hidden_state=latent_out_copy)
            cls_loss = cls_criterion(classifier_out, trg_label)
            cls_model.zero_grad()
            cls_loss.backward()
            latent_out_copy_grad = latent_out_copy.grad.data
            if epoch == 1:
                print(latent_out_copy_grad)

            for epsilon in [2, 5, 8]:
                latent_out_copy = latent_out_copy - ((epsilon * 0.9) * latent_out_copy_grad)

                with torch.no_grad():
                    recon_out = model(input_ids=src_sequence, attention_mask=src_att, encoder_out=encoder_out, latent_out=latent_out_copy)

                # Augmenting
                eps_dict[f'eps_{epsilon}'] = model.tokenizer.batch_decode(recon_out.argmax(dim=2), skip_special_tokens=True)[0]

                with torch.no_grad():
                    inp_dict = model.tokenizer(eps_dict[f'eps_{epsilon}'],
                                                   max_length=args.src_max_len,
                                                   padding='max_length',
                                                   truncation=True,
                                                   return_tensors='pt')
                    encoder_out = model.encode(input_ids=inp_dict['input_ids'].to(device), attention_mask=inp_dict['attention_mask'].to(device))
                    latent_out, _ = model.latent_encode(encoder_out=encoder_out)
                    classifier_out = cls_model(hidden_state=latent_out)
                    prob_dict[f'eps_{epsilon}'] = F.softmax(classifier_out)

            write_log(logger, f'Generated Examples')
            write_log(logger, f'Phase: {phase}')
            write_log(logger, f'Source: {src_output}')
            write_log(logger, f'Source Probability: {prob_dict["eps_0"]}')
            write_log(logger, f'Augmented_origin: {eps_dict["eps_0"]}')
            write_log(logger, f'Augmented_2: {eps_dict["eps_2"]}')
            write_log(logger, f'Augmented_2_prob: {prob_dict["eps_2"]}')
            write_log(logger, f'Augmented_5: {eps_dict["eps_5"]}')
            write_log(logger, f'Augmented_5_prob: {prob_dict["eps_5"]}')
            write_log(logger, f'Augmented_8: {eps_dict["eps_8"]}')
            write_log(logger, f'Augmented_8_prob: {prob_dict["eps_8"]}')

    # 3) Results
    # write_log(logger, f'Best AUG Epoch: {best_aug_epoch}')
    # write_log(logger, f'Best AUG Loss: {round(best_aug_val_loss.item(), 2)}')
    write_log(logger, f'Best CLS Epoch: {best_cls_epoch}')
    write_log(logger, f'Best CLS Loss: {round(best_cls_val_loss.item(), 2)}')