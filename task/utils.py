import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from datasets import load_dataset

def data_split_index(seq, valid_ratio: float = 0.1, test_ratio: float = 0.03):

    paired_data_len = len(seq)
    valid_num = int(paired_data_len * valid_ratio)
    test_num = int(paired_data_len * test_ratio)

    valid_index = np.random.choice(paired_data_len, valid_num, replace=False)
    train_index = list(set(range(paired_data_len)) - set(valid_index))
    test_index = np.random.choice(train_index, test_num, replace=False)
    train_index = list(set(train_index) - set(test_index))

    return train_index, valid_index, test_index

def data_load(args):

    src_list = dict()
    trg_list = dict()

    if args.data_name == 'IMDB':
        imdb_data_path = os.path.join(args.data_path,'IMDB')

        train_dat = pd.read_csv(os.path.join(imdb_data_path, 'train.csv'))
        test_dat = pd.read_csv(os.path.join(imdb_data_path, 'test.csv'))

        train_dat['sentiment'] = train_dat['sentiment'].replace('positive', 0)
        train_dat['sentiment'] = train_dat['sentiment'].replace('negative', 1)
        test_dat['sentiment'] = test_dat['sentiment'].replace('positive', 0)
        test_dat['sentiment'] = test_dat['sentiment'].replace('negative', 1)

        train_index, valid_index, test_index = data_split_index(train_dat, valid_ratio=args.valid_ratio, test_ratio=0)

        src_list['train'] = [train_dat['comment'].tolist()[i] for i in train_index]
        trg_list['train'] = [train_dat['sentiment'].tolist()[i] for i in train_index]

        src_list['valid'] = [train_dat['comment'].tolist()[i] for i in valid_index]
        trg_list['valid'] = [train_dat['sentiment'].tolist()[i] for i in valid_index]

        src_list['test'] = test_dat['comment'].tolist()
        trg_list['test'] = test_dat['sentiment'].tolist()

    return src_list, trg_list

def input_to_device(batch_iter, aug_batch_iter, device):
    src_sequence = batch_iter[0]
    src_att = batch_iter[1]
    src_seg = batch_iter[2]
    trg_label = batch_iter[3]

    src_sequence = src_sequence.to(device, non_blocking=True)
    src_att = src_att.to(device, non_blocking=True)
    src_seg = src_seg.to(device, non_blocking=True)
    trg_label = trg_label.to(device, non_blocking=True)

    # Augmentation input setting
    aug_src_sequence = aug_batch_iter[0]
    aug_src_att = aug_batch_iter[1]
    aug_src_seg = aug_batch_iter[2]
    aug_trg_label = aug_batch_iter[3]

    aug_src_sequence = aug_src_sequence.to(device, non_blocking=True)
    aug_src_att = aug_src_att.to(device, non_blocking=True)
    aug_src_seg = aug_src_seg.to(device, non_blocking=True)
    aug_trg_label = aug_trg_label.to(device, non_blocking=True)
    
    return (src_sequence, src_att, src_seg, trg_label), (aug_src_sequence, aug_src_att, aug_src_seg, aug_trg_label)