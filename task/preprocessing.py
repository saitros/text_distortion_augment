import os
import time
import h5py
import pickle
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
# Import custom modules
from task.utils import data_load, data_sampling, tokenizing
from utils import TqdmLoggingHandler, write_log, return_model_name

from datasets import load_dataset

def preprocessing(args):

    start_time = time.time()

    #===================================#
    #==============Logging==============#
    #===================================#

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False

    #===================================#
    #=============Data Load=============#
    #===================================#

    write_log(logger, 'Start preprocessing!')

    src_list, trg_list = data_load(args)
    src_list, trg_list = data_sampling(args, src_list, trg_list)
    write_log(logger, 'Data loading done! ; {0}min spend'.format(round((time.time()-start_time)/60, 3)))

    #===================================#
    #==========Pre-processing===========#
    #===================================#

    write_log(logger, 'Tokenizer setting...')
    start_time = time.time()

    model_name = return_model_name(args.encoder_model_type)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    processed_sequences = tokenizing(args, src_list, tokenizer)

    write_log(logger, f'Done! ; {round((time.time()-start_time)/60, 3)}min spend')

    #===================================#
    #==============Saving===============#
    #===================================#

    write_log(logger, 'Parsed sentence saving...')
    start_time = time.time()

    # Path checking
    save_path = os.path.join(args.preprocess_path, args.data_name, args.encoder_model_type)

    with h5py.File(os.path.join(save_path, f'src_len_{args.src_max_len}_processed.hdf5'), 'w') as f:
        f.create_dataset('train_src_input_ids', data=processed_sequences['train']['input_ids'])
        f.create_dataset('train_src_attention_mask', data=processed_sequences['train']['attention_mask'])
        f.create_dataset('valid_src_input_ids', data=processed_sequences['valid']['input_ids'])
        f.create_dataset('valid_src_attention_mask', data=processed_sequences['valid']['attention_mask'])
        f.create_dataset('train_label', data=np.array(trg_list['train']).astype(int))
        f.create_dataset('valid_label', data=np.array(trg_list['valid']).astype(int))
        if args.encoder_model_type == 'bert':
            f.create_dataset('train_src_token_type_ids', data=processed_sequences['train']['token_type_ids'])
            f.create_dataset('valid_src_token_type_ids', data=processed_sequences['valid']['token_type_ids'])

    with h5py.File(os.path.join(save_path, f'src_len_{args.src_max_len}_test_processed.hdf5'), 'w') as f:
        f.create_dataset('test_src_input_ids', data=processed_sequences['test']['input_ids'])
        f.create_dataset('test_src_attention_mask', data=processed_sequences['test']['attention_mask'])
        f.create_dataset('test_label', data=np.array(trg_list['test']).astype(int))
        if args.encoder_model_type == 'bert':
            f.create_dataset('test_src_token_type_ids', data=processed_sequences['test']['token_type_ids'])

    # Word2id pickle file save
    word2id_dict = {
        'src_word2id' : tokenizer.get_vocab(),
        'num_labels': len(set(trg_list['train']))
    }

    with open(os.path.join(save_path, 'word2id.pkl'), 'wb') as f:
        pickle.dump(word2id_dict, f)

    write_log(logger, f'Done! ; {round((time.time()-start_time)/60, 3)}min spend')