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