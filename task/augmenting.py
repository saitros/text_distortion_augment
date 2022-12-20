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
from model.model import AugModel
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
        train_src_attention_mask = f.get('train_src_attention_mask')[:
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
    model = AugModel(encoder_model_type='bert', decoder_model_type='bert',
                     isPreTrain=args.isPreTrain, z_variation=args.z_variation,
                     src_max_len=args.src_max_len, dropout=args.dropout)
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
    write_log(logger, f"Total number of trainingsets  iterations - {len(dataset_dict['train'])}, {len(dataloader_dict['train'])}")
    
    # 3) Model loading
    cudnn.benchmark = True
    save_file_name = os.path.join(args.model_save_path, args.data_name, args.model_type, 'checkpoint.pth.tar')
    checkpoint = torch.load(save_file_name)
    aug_model.load_state_dict(checkpoint['aug_model'])
    cls_model.load_state_dict(checkpoint['cls_model'])
    write_log(logger, f'Loaded model from {save_file_name}')

    #===================================#
    #=========Model Train Start=========#
    #===================================#
    
    start_time_e = time()
    write_log(logger, 'Augmenting start...')
    model.eval()
    seq_list = list()
    aug_list = list()

    for i, batch_iter in enumerate(tqdm(dataloader_dict['train'], bar_format='{l_bar}{bar:30}{r_bar}{bar:-2b}')):

        # Input setting
        b_iter = input_to_device(batch_iter, device=device)
        src_sequence, src_att, src_seg, trg_label = b_iter

        # Augmenter
        with torch.no_grad():
            decoder_out, encoder_out, latent_out = aug_model(src_input_ids=src_sequence, 
                                                                src_attention_mask=src_att)
        encoder_out_copy = encoder_out.clone().detach().requires_grad_(True)
        latent_out_copy = latent_out.clone().detach().requires_grad_(True)

        for epsilon in [2.0, 3.0, 4.0, 5.0]: # * 0.9
            data = Variable(encoder_out_copy + latent_out_copy.unsqueeze(1), volatile=False)
            data.requires_grad = True

            logit = cls_model(encoder_out=data)
            cls_loss = cls_criterion(logit, trg_label)
            cls_model.zero_grad()
            cls_loss.backward()
            data_grad = data.grad
            data = data - epsilon * data_grad

            with torch.no_grad():
                with autocast():
                    decoder_out = aug_model.generate(src_input_ids=src_sequence, 
                                                     src_attention_mask=src_att,
                                                     z=data)

            # Augmenting
            augmented_output = aug_model.tokenizer.batch_decode(decoder_out, skip_special_tokens=True)
            src_output = aug_model.tokenizer.batch_decode(src_sequence, skip_special_tokens=True)

    result_dat = pd.DataFrame({
        'seq': seq_list,
        'aug': aug_list,
        'aug2': aug2_list
    })
    result_dat.to_csv('./result.csv', index=False)