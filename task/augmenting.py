# Import modules
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import gc
import h5py
import pickle
import logging
import numpy as np
from tqdm import tqdm
from time import time
from transformers import AutoTokenizer
# Import PyTorch
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
# Import custom modules
from model.model import AugModel
from model.dataset import CustomDataset
from utils import TqdmLoggingHandler, write_log

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

    save_path = os.path.join(args.preprocess_path, args.data_name)

    with h5py.File(os.path.join(save_path, 'processed.hdf5'), 'r') as f:
        train_src_input_ids = f.get('train_src_input_ids')[:]
        train_src_attention_mask = f.get('train_src_attention_mask')[:]
        train_src_token_type_ids = f.get('train_src_token_type_ids')[:]
        train_trg_list = f.get('train_label')[:]
        train_trg_list = F.one_hot(torch.tensor(train_trg_list, dtype=torch.long)).numpy()

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
    save_file_name = os.path.join(args.model_save_path, args.data_name)
    save_file_name += 'checkpoint.pth.tar'
    checkpoint = torch.load(save_file_name)
    model.load_state_dict(checkpoint['model'])

    #===================================#
    #=========Model Train Start=========#
    #===================================#

    write_log(logger, 'Traing start!')
    best_val_loss = 1e+10
    
    start_time_e = time()
    write_log(logger, 'Augmenting start...')
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    aug_list = list()

    for i, batch_iter in enumerate(tqdm(dataloader_dict['train'], bar_format='{l_bar}{bar:30}{r_bar}{bar:-2b}')):

        # Input setting
        src_sequence = batch_iter[0]
        src_att = batch_iter[1]
        src_seg = batch_iter[2]
        trg_label = batch_iter[3]

        src_sequence = src_sequence.to(device, non_blocking=True)
        src_att = src_att.to(device, non_blocking=True)
        src_seg = src_seg.to(device, non_blocking=True)
        trg_label = trg_label.to(device, non_blocking=True)

        # Reconsturction setting
        non_pad = src_sequence != model.pad_idx

        with torch.no_grad():
            with autocast():
                encoder_out, decoder_out, z = model(src_input_ids=src_sequence, 
                                                    src_attention_mask=src_att,
                                                    src_token_type_ids=src_seg,
                                                    non_pad_position=non_pad)
            
        # 
        print(tokenizer.batch_decode(decoder_out.argmax(dim=1)))
        aug_list.append(tokenizer.batch_decode(decoder_out.argmax(dim=1)))