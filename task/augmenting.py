# Import modules
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import gc
import h5py
import pickle
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time
from transformers import AutoTokenizer
# Import PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
# Import custom modules
from model.model import TransformerModel, ClassifierModel
from model.dataset import CustomDataset
from utils import TqdmLoggingHandler, write_log
from task.utils import input_to_device

def augmenting(args):
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
        train_trg_list = f.get('train_label')[:]
        train_trg_list = F.one_hot(torch.tensor(train_trg_list, dtype=torch.long)).numpy()
        if args.model_type == 'bert':
            train_src_token_type_ids = f.get('train_src_token_type_ids')[:]
        else:
            train_src_token_type_ids = list()

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
    model.to(device)

    # 2) Dataloader setting
    dataset_dict = {
        'train': CustomDataset(src_list=train_src_input_ids, src_att_list=train_src_attention_mask,
                               src_seg_list=train_src_token_type_ids,
                               trg_list=train_trg_list, src_max_len=args.src_max_len)
    }
    dataloader_dict = {
        'train': DataLoader(dataset_dict['train'], drop_last=False,
                            batch_size=args.batch_size, shuffle=False, pin_memory=True,
                            num_workers=args.num_workers)
    }
    write_log(logger, f"Total number of trainingsets iterations - {len(dataset_dict['train'])}, {len(dataloader_dict['train'])}")

    # 3) Model loading
    cudnn.benchmark = True
    cls_criterion = nn.CrossEntropyLoss().to(device)
    save_file_name = os.path.join(args.model_save_path, args.data_name, args.model_type, f'{args.cls_scheduler}_{args.aug_num_epochs}_checkpoint.pth.tar')
    checkpoint = torch.load(save_file_name)
    model.load_state_dict(checkpoint['model'])
    cls_model.load_state_dict(checkpoint['cls_model'])
    write_log(logger, f'Loaded augmenter model from {save_file_name}')

    #===================================#
    #=========Model Train Start=========#
    #===================================#
    
    start_time_e = time()
    write_log(logger, 'Augmenting start...')
    model.eval()
    # cls_model.eval()
    # seq_list = list()
    # aug_list = dict()
    # epsilon_list = [0, 1, 2, 3, 5, 8, 10]
    # latent_pre_list = list()
    # latent_now_list = list()
    # latent_next_list = list()

    # for i in epsilon_list:
    #     aug_list[f'eps_{i}'] = list()
    #     aug_list[f'eps_{i}_prob'] = list()

    for i, batch_iter in enumerate(tqdm(dataloader_dict['train'], bar_format='{l_bar}{bar:30}{r_bar}{bar:-2b}')):

        # Input setting
        b_iter = input_to_device(batch_iter, device=device)
        src_sequence, src_att, src_seg, trg_label = b_iter
        trg_fliped_label = torch.flip(trg_label, dims=[1])

        # Source De-tokenizing
        origin_source = model.tokenizer.batch_decode(src_sequence, skip_special_tokens=True)
        origin_list.extend(origin_source)
        
        # Encoding
        with torch.no_grad():
            encoder_out = model.encode(input_ids=src_sequence, attention_mask=src_att)
            recon_out = model.generate(encoder_out=encoder_out, attention_mask=src_att, beam_size=args.beam_size,
                                       beam_alpha=args.beam_alpha, repetition_penalty=args.repetition_penalty, device=device)
            beam_output = model.tokenizer.batch_decode(recon_out, skip_special_tokens=True)

            inp_dict = model.tokenizer(beam_output,
                                        max_length=args.src_max_len,
                                        padding='max_length',
                                        truncation=True,
                                        return_tensors='pt')

        # Classification for FGSM
        encoder_out_copy = encoder_out.clone().detach().requires_grad_(True)
        classifier_out = model.classify(hidden_states=encoder_out_copy)

        # Save results
        aug_dict['eps_0'].extend(beam_output)
        prob_dict['eps_0'].extend(F.softmax(classifier_out, dim=1))

        # 
        cls_loss = cls_criterion(classifier_out, trg_fliped_label)
        model.zero_grad()
        cls_loss.backward()
        encoder_out_copy_grad = encoder_out_copy.grad.data
        encoder_out_copy_grad_sign = encoder_out_copy_grad.sign()

        for step in range(51): # * 0.9
            
            epsilon = 0.8 # Need to fix

            encoder_out_copy = encoder_out_copy + (epsilon * encoder_out_copy_grad_sign)

            with torch.no_grad():
                recon_out = model.generate(encoder_out=encoder_out_copy, attention_mask=src_att, beam_size=args.beam_size,
                                        beam_alpha=args.beam_alpha, repetition_penalty=args.repetition_penalty, device=device)
                beam_output = model.tokenizer.batch_decode(recon_out, skip_special_tokens=True)

                inp_dict = model.tokenizer(beam_output,
                                            max_length=args.src_max_len,
                                            padding='max_length',
                                            truncation=True,
                                            return_tensors='pt')

            # Classification for FGSM
            encoder_out_copy = encoder_out.clone().detach().requires_grad_(True)
            classifier_out = model.classify(hidden_states=encoder_out_copy)

            # Save results
            if step in [10, 20, 30, 40, 50]:
                aug_dict[f'eps_{step}'].extend(beam_output)
                prob_dict[f'eps_{step}'].extend(F.softmax(classifier_out, dim=1))

            # 
            cls_loss = cls_criterion(classifier_out, trg_fliped_label)
            model.zero_grad()
            cls_loss.backward()
            encoder_out_copy_grad = encoder_out_copy.grad.data
            encoder_out_copy_grad_sign = encoder_out_copy_grad.sign()

        if args.debuging_mode:
            break
        
    result_dat = pd.DataFrame({
        'origin': origin_list,
        'eps_0': aug_dict['eps_0'],
        'eps_0_prob': prob_dict['eps_0']
    })
    for idx in [10, 20, 30, 40, 50]:
        result_dat[f'eps_{idx}'] = aug_dict[f'eps_{idx}']
        result_dat[f'eps_{idx}_prob'] = prob_dict[f'eps_{idx}']

    result_dat.to_csv(f'./augmenting_result.csv', index=False)