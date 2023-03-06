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
from model.model import TransformerModel
from model.dataset import CustomDataset
from model.loss import compute_mmd, CustomLoss
from optimizer.utils import shceduler_select, optimizer_select
from optimizer.scheduler import get_cosine_schedule_with_warmup
from utils import TqdmLoggingHandler, write_log, get_tb_exp_name
from task.utils import input_to_device

def augmenter_training(args):
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

    save_path = os.path.join(args.preprocess_path, args.data_name, args.encoder_model_type)

    with h5py.File(os.path.join(save_path, f'src_len_{args.src_max_len}_processed.hdf5'), 'r') as f:
        train_src_input_ids = f.get('train_src_input_ids')[:]
        train_src_attention_mask = f.get('train_src_attention_mask')[:]
        valid_src_input_ids = f.get('valid_src_input_ids')[:]
        valid_src_attention_mask = f.get('valid_src_attention_mask')[:]
        train_trg_list = f.get('train_label')[:]
        train_trg_list = F.one_hot(torch.tensor(train_trg_list, dtype=torch.long)).numpy()
        valid_trg_list = f.get('valid_label')[:]
        valid_trg_list = F.one_hot(torch.tensor(valid_trg_list, dtype=torch.long)).numpy()
        if args.encoder_model_type == 'bert':
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
    model = TransformerModel(encoder_model_type=args.encoder_model_type, decoder_model_type=args.decoder_model_type, 
                             isPreTrain=args.isPreTrain, encoder_out_mix_ratio=args.encoder_out_mix_ratio,
                             encoder_out_cross_attention=args.encoder_out_cross_attention,
                             encoder_out_to_augmenter=args.encoder_out_to_augmenter, classify_method=args.classify_method,
                             src_max_len=args.src_max_len, num_labels=num_labels, dropout=args.dropout)
    model.to(device)

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
                            batch_size=args.batch_size, shuffle=True, pin_memory=True,
                            num_workers=args.num_workers)
    }
    write_log(logger, f"Total number of trainingsets  iterations - {len(dataset_dict['train'])}, {len(dataloader_dict['train'])}")

    del (
        train_src_input_ids, train_src_attention_mask, train_src_token_type_ids, train_trg_list,
        valid_src_input_ids, valid_src_attention_mask, valid_src_token_type_ids, valid_trg_list
    )

    # 3) Optimizer & Learning rate scheduler setting
    cls_optimizer = optimizer_select(optimizer_model=args.cls_optimizer, model=model, lr=args.cls_lr, w_decay=args.w_decay)
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
        save_file_name = os.path.join(args.model_save_path, args.data_name, args.encoder_model_type, 'checkpoint.pth.tar')
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

        for i, batch_iter in enumerate(tqdm(dataloader_dict['train'], bar_format='{l_bar}{bar:30}{r_bar}{bar:-2b}')):

            cls_optimizer.zero_grad(set_to_none=True)

            # Input setting
            b_iter = input_to_device(batch_iter, device=device)
            src_sequence, src_att, src_seg, trg_label = b_iter

            # Encoding
            encoder_out = model.encode(input_ids=src_sequence, attention_mask=src_att)
            hidden_states = encoder_out
            if args.classify_method == 'latent_out':
                latent_out, latent_encoder_out = model.latent_encode(encoder_out=encoder_out)
                hidden_states = latent_out

            # Classifier
            classifier_out = model.classify(hidden_states=hidden_states)
            cls_loss = cls_criterion(classifier_out, trg_label)
            if args.latent_mmd_loss:
                mmd_loss = compute_mmd(latent_encoder_out, z_var=args.z_variation) * 100
            else:
                mmd_loss = torch.tensor(0)

            # Loss Backward
            total_cls_loss = cls_loss + mmd_loss
            total_cls_loss.backward()
            if args.clip_grad_norm > 0:
                clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            cls_optimizer.step()
            cls_scheduler.step()

            # Print loss value only training
            if i == 0 or i % args.print_freq == 0 or i == len(dataloader_dict['train'])-1:
                train_acc = (classifier_out.argmax(dim=1) == trg_label.argmax(dim=1)).sum() / len(trg_label)
                iter_log = "[Epoch:%03d][%03d/%03d] train_cls_loss:%03.2f | train_accuracy:%03.2f | train_mmd_loss:%03.2f | learning_rate:%1.6f |spend_time:%02.2fmin" % \
                    (epoch, i, len(dataloader_dict['train'])-1, cls_loss.item(), train_acc.item(), mmd_loss.item(), cls_optimizer.param_groups[0]['lr'], (time() - start_time_e) / 60)
                write_log(logger, iter_log)

            if args.debuging_mode:
                break

        write_log(logger, 'Classifier validation start...')
        model.eval()
        val_cls_loss = 0
        val_acc = 0
        val_mmd_loss = 0

        for batch_iter in tqdm(dataloader_dict['valid'], bar_format='{l_bar}{bar:30}{r_bar}{bar:-2b}'):

            b_iter = input_to_device(batch_iter, device=device)
            src_sequence, src_att, src_seg, trg_label = b_iter

            with torch.no_grad():
                encoder_out = model.encode(input_ids=src_sequence, attention_mask=src_att)
                hidden_states = encoder_out
                if args.classify_method == 'latent_out':
                    latent_out, latent_encoder_out = model.latent_encode(encoder_out=encoder_out)
                    hidden_states = latent_out

                # Classifier
                classifier_out = model.classify(hidden_states=hidden_states)
                cls_loss = cls_criterion(classifier_out, trg_label)
                if args.latent_mmd_loss:
                    mmd_loss = compute_mmd(latent_encoder_out, z_var=args.z_variation) * 100
                else:
                    mmd_loss = torch.tensor(0)

            # Loss and Accuracy Check
            val_acc += (classifier_out.argmax(dim=1) == trg_label.argmax(dim=1)).sum() / len(trg_label)
            val_cls_loss += cls_loss
            val_mmd_loss += mmd_loss

            if args.debuging_mode:
                break

        # val_mmd_loss /= len(dataloader_dict['valid'])
        val_cls_loss /= len(dataloader_dict['valid'])
        val_acc /= len(dataloader_dict['valid'])
        write_log(logger, 'Classifier Validation MMD Loss: %3.3f' % val_mmd_loss)
        write_log(logger, 'Classifier Validation CrossEntropy Loss: %3.3f' % val_cls_loss)
        write_log(logger, 'Classifier Validation Accuracy: %3.2f%%' % (val_acc * 100))

        save_file_name = os.path.join(args.model_save_path, args.data_name, args.encoder_model_type, 'checkpoint.pth.tar')
        if val_cls_loss < best_cls_val_loss:
            write_log(logger, 'Model checkpoint saving...')
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

    for para in model.encoder.parameters():
        para.requires_grad = False
    for para in model.latent_encoder.parameters():
        para.requires_grad = False
    for para in model.latent_decoder.parameters():
        para.requires_grad = False
    for para in model.classifier1.parameters():
        para.requires_grad = False
    for para in model.classifier1.parameters():
        para.requires_grad = False
    for para in model.classifier2.parameters():
        para.requires_grad = False
    for para in model.classifier3.parameters():
        para.requires_grad = False

    for epoch in range(start_epoch + 1, args.aug_num_epochs + 1):
        start_time_e = time()

        write_log(logger, 'Augmenter training start...')
        model.train()

        for i, batch_iter in enumerate(tqdm(dataloader_dict['train'], bar_format='{l_bar}{bar:30}{r_bar}{bar:-2b}')):

            aug_optimizer.zero_grad(set_to_none=True)

            # Input setting
            b_iter = input_to_device(batch_iter, device=device)
            src_sequence, src_att, src_seg, trg_label = b_iter

            # Encoding
            with torch.no_grad():
                encoder_out = model.encode(input_ids=src_sequence, attention_mask=src_att)
                latent_out = None
                hidden_states = encoder_out
                if args.encoder_out_mix_ratio != 0:
                    latent_out, latent_encoder_out = model.latent_encode(encoder_out=encoder_out)
                    hidden_states = latent_out

            # Classifier Results
            with torch.no_grad():
                classifier_out = model.classify(hidden_states=hidden_states)

            # Reconstruction
            recon_out = model(input_ids=src_sequence, attention_mask=src_att, encoder_out=encoder_out, latent_out=latent_out)
            recon_loss = recon_criterion(recon_out.view(-1, src_vocab_num), src_sequence.contiguous().view(-1))

            # Additional Loss
            encoder_out = model.encode(input_ids=recon_out)
            latent_out = None
            hidden_states = encoder_out
            if args.encoder_out_mix_ratio != 0:
                latent_out, latent_encoder_out = model.latent_encode(encoder_out=encoder_out)
                hidden_states = latent_out

            re_classifier_out = model.classify(hidden_states=hidden_states)
            cls_loss2 = cls_criterion(re_classifier_out, classifier_out)

            # Loss Backward
            total_loss = recon_loss + cls_loss2
            total_loss.backward()
            if args.clip_grad_norm > 0:
                clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            aug_optimizer.step()
            aug_scheduler.step()

            # Print loss value only training
            if i == 0 or i % args.print_freq == 0 or i == len(dataloader_dict['train'])-1:
                # train_acc = (recon_out.view(-1, src_vocab_num).argmax(dim=1) == src_sequence.contiguous().view(-1)).sum() / len(src_sequence)
                iter_log = "[Epoch:%03d][%03d/%03d] train_recon_loss:%03.2f | train_cls2_loss:%03.2f | learning_rate:%1.6f |spend_time:%02.2fmin" % \
                    (epoch, i, len(dataloader_dict['train'])-1, recon_loss.item(), cls_loss2.item(), aug_optimizer.param_groups[0]['lr'], (time() - start_time_e) / 60)
                write_log(logger, iter_log)

            if args.debuging_mode:
                break

        #===================================#
        #============Validation=============#
        #===================================#
        write_log(logger, 'Augmenter validation start...')

        # Validation
        model.eval()
        val_recon_loss = 0

        for batch_iter in tqdm(dataloader_dict['valid'], bar_format='{l_bar}{bar:30}{r_bar}{bar:-2b}'):

            b_iter = input_to_device(batch_iter, device=device)
            src_sequence, src_att, src_seg, trg_label = b_iter

            # Encoding
            with torch.no_grad():
                encoder_out = model.encode(input_ids=src_sequence, attention_mask=src_att)
                latent_out = None
                if args.encoder_out_mix_ratio != 0:
                    latent_out, latent_encoder_out = model.latent_encode(encoder_out=encoder_out)

                # Reconstruction
                recon_out = model(input_ids=src_sequence, attention_mask=src_att, encoder_out=encoder_out, latent_out=latent_out)

                recon_loss = recon_criterion(recon_out.view(-1, src_vocab_num), src_sequence.contiguous().view(-1))
                val_recon_loss += recon_loss

            if args.debuging_mode:
                break

        val_recon_loss /= len(dataloader_dict['valid'])
        write_log(logger, 'Augmenter Validation CrossEntropy Loss: %3.3f' % val_recon_loss)

        save_file_name = os.path.join(args.model_save_path, args.data_name, args.encoder_model_type, 'checkpoint.pth.tar')
        if val_recon_loss < best_aug_val_loss:
            write_log(logger, 'Model checkpoint saving...')
            torch.save({
                'cls_training_done': True,
                'epoch': epoch,
                'model': model.state_dict(),
                'cls_optimizer': cls_optimizer.state_dict(),
                'aug_optimizer': aug_optimizer.state_dict(),
                'cls_scheduler': cls_scheduler.state_dict(),
                'aug_scheduler': aug_scheduler.state_dict(),
            }, save_file_name)
            best_aug_val_loss = val_recon_loss
            best_aug_epoch = epoch
        else:
            else_log = f'Still {best_aug_epoch} epoch Loss({round(best_aug_val_loss.item(), 2)}) is better...'
            write_log(logger, else_log)

        #===================================#
        #=========Text Augmentation=========#
        #===================================#

        eps_dict, prob_dict = dict(), dict()

        for para in model.encoder.parameters():
            para.requires_grad = True
        for para in model.latent_encoder.parameters():
            para.requires_grad = True
        for para in model.latent_decoder.parameters():
            para.requires_grad = True
        for para in model.classifier1.parameters():
            para.requires_grad = True
        for para in model.classifier1.parameters():
            para.requires_grad = True
        for para in model.classifier2.parameters():
            para.requires_grad = True
        for para in model.classifier3.parameters():
            para.requires_grad = True

        for phase in ['train', 'valid']:
            example_iter = next(iter(dataloader_dict[phase]))

            example_iter_ = input_to_device(example_iter, device=device)
            src_sequence, src_att, src_seg, trg_label = example_iter_

            src_output = model.tokenizer.batch_decode(src_sequence, skip_special_tokens=True)[0]

            # Target Label Setting
            if args.label_flipping:
                fliped_trg_label = torch.flip(trg_label, dims=[1]).to(device)
            else:
                fliped_trg_label = torch.ones_like(trg_label) * 0.5

            # Forward
            with torch.no_grad():
                encoder_out = model.encode(input_ids=src_sequence, attention_mask=src_att)
                hidden_states = encoder_out
                if args.classify_method == 'latent_out':
                    latent_out, latent_encoder_out = model.latent_encode(encoder_out=encoder_out)
                    hidden_states = latent_out

                # Classifier
                classifier_out = model.classify(hidden_states=hidden_states)
                src_output_prob = F.softmax(classifier_out)[0]

                # Reconstruction
                latent_out = None
                if args.encoder_out_mix_ratio != 0:
                    latent_out, latent_encoder_out = model.latent_encode(encoder_out=encoder_out)
                recon_out = model(input_ids=src_sequence, attention_mask=src_att, encoder_out=encoder_out, latent_out=latent_out)

            # Output Tokenizing & Pre-processing
            detokenized = model.tokenizer.batch_decode(recon_out.argmax(dim=2), skip_special_tokens=True)
            eps_dict['eps_0'] = detokenized[0]
            inp_dict = model.tokenizer(detokenized, max_length=args.src_max_len,
                                       padding='max_length', truncation=True, return_tensors='pt')

            # Probability Calculate
            with torch.no_grad():
                encoder_out_eps_0 = model.encode(input_ids=inp_dict['input_ids'].to(device),
                                                 attention_mask=inp_dict['attention_mask'].to(device))
                hidden_states = encoder_out_eps_0

            if args.classify_method == 'latent_out':
                latent_out_eps_0, _ = model.latent_encode(encoder_out=encoder_out_eps_0)
                hidden_states = latent_out_eps_0

            hidden_states_grad_true = hidden_states.clone().detach().requires_grad_(True)
            classifier_out = model.classify(hidden_states=hidden_states_grad_true)
            prob_dict['eps_0'] = F.softmax(classifier_out)[0]

            for _ in range(3):

                model.zero_grad()
                cls_loss = cls_criterion(classifier_out, fliped_trg_label)
                cls_loss.backward()
                hidden_states_grad = hidden_states_grad_true.grad.data.sign()
                
                # Gradient-based Modification
                if args.classify_method == 'latent_out':
                    encoder_out_copy = encoder_out.clone().detach()
                    latent_out_copy = hidden_states_grad_true - (args.grad_epsilon * hidden_states_grad)
                    hidden_states_grad_true = latent_out_copy.clone()
                else:
                    encoder_out_copy = hidden_states_grad_true - (args.grad_epsilon * hidden_states_grad)
                    latent_out_copy = None
                    hidden_states_grad_true = encoder_out_copy.clone()

                hidden_states_grad_true = hidden_states_grad_true.clone().detach().requires_grad_(True)
                classifier_out = model.classify(hidden_states=hidden_states_grad_true)
                revised_prob = F.softmax(classifier_out)[0]

            with torch.no_grad():
                recon_out = model(input_ids=src_sequence, attention_mask=src_att, encoder_out=encoder_out_copy, latent_out=latent_out_copy)
                recon_decoding = model.tokenizer.batch_decode(recon_out.argmax(dim=2), skip_special_tokens=True)
                eps_dict['eps_1'] = recon_decoding[0]

            inp_dict = model.tokenizer(recon_decoding,
                                        max_length = args.src_max_len,
                                        padding='max_length',
                                        truncation=True,
                                        return_tensors='pt')

            # Probability Calculate
            with torch.no_grad():
                encoder_out_eps_1 = model.encode(input_ids=inp_dict['input_ids'].to(device),
                                                 attention_mask=inp_dict['attention_mask'].to(device))
                hidden_states = encoder_out_eps_1

            if args.classify_method == 'latent_out':
                latent_out_eps_1, _ = model.latent_encode(encoder_out=encoder_out_eps_1)
                hidden_states = latent_out_eps_1

            hidden_states_grad_true = hidden_states.clone().detach().requires_grad_(True)
            classifier_out = model.classify(hidden_states=hidden_states_grad_true)
            prob_dict['eps_1'] = F.softmax(classifier_out)[0]

            dict_key = prob_dict.keys()

            write_log(logger, f'Generated Examples')
            write_log(logger, f'Epsilon repeat: 3')
            write_log(logger, f'Phase: {phase}')
            write_log(logger, f'Source: {src_output}')
            write_log(logger, f'Source Probability: {src_output_prob}')
            write_log(logger, f'Reconstructed: {eps_dict["eps_0"]}')
            write_log(logger, f'Reconstructed Probability: {prob_dict["eps_0"]}')
            write_log(logger, '-'*50)
            write_log(logger, f'Revised Probability: {revised_prob}')
            write_log(logger, f'Augmented: {eps_dict["eps_1"]}')
            write_log(logger, f'Augmented Probability: {prob_dict["eps_1"]}')

    # 3) Results
    write_log(logger, f'Best AUG Epoch: {best_aug_epoch}')
    write_log(logger, f'Best AUG Loss: {round(best_aug_val_loss.item(), 2)}')
    write_log(logger, f'Best CLS Epoch: {best_cls_epoch}')
    write_log(logger, f'Best CLS Loss: {round(best_cls_val_loss.item(), 2)}')

# 1. 단순 reconstruction때는 [0,1]로 문장 나오면 되고
# 2. augment 한거는 옮겨진 만큼으로 나오면 되고
# 1 학습 하고 2 학습 하고