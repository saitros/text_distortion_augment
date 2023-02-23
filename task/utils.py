import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm

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
        dataset = load_dataset("imdb")

        train_dat = dataset['train']
        test_dat = dataset['test']

        train_index, valid_index, test_index = data_split_index(train_dat, valid_ratio=args.valid_ratio, test_ratio=0)

        src_list['train'] = [train_dat['text'][i] for i in tqdm(train_index)]
        trg_list['train'] = [train_dat['label'][i] for i in tqdm(train_index)]

        src_list['valid'] = [train_dat['text'][i] for i in tqdm(valid_index)]
        trg_list['valid'] = [train_dat['label'][i] for i in tqdm(valid_index)]

        src_list['test'] = test_dat['text']
        trg_list['test'] = test_dat['label']

    if args.data_name == 'sst2':
        dataset = load_dataset("glue", args.data_name)
        train_dat = pd.DataFrame(dataset['train'])
        valid_dat = pd.DataFrame(dataset['validation'])
        test_dat = pd.DataFrame(dataset['test'])

        src_list['train'] = train_dat['sentence'].tolist()
        trg_list['train'] = train_dat['label'].tolist()

        src_list['valid'] = valid_dat['sentence'].tolist()
        trg_list['valid'] = valid_dat['label'].tolist()

        src_list['test'] = test_dat['sentence'].tolist()
        trg_list['test'] = test_dat['label'].tolist()

    if args.data_name == 'cola':
        dataset = load_dataset("glue", args.data_name)
        train_dat = pd.DataFrame(dataset['train'])
        valid_dat = pd.DataFrame(dataset['validation'])
        test_dat = pd.DataFrame(dataset['test'])

        src_list['train'] = train_dat['sentence'].tolist()
        trg_list['train'] = train_dat['label'].tolist()

        src_list['valid'] = valid_dat['sentence'].tolist()
        trg_list['valid'] = valid_dat['label'].tolist()

        src_list['test'] = test_dat['sentence'].tolist()
        trg_list['test'] = test_dat['label'].tolist()

    if args.data_name == 'mnli':
        dataset = load_dataset("glue", args.data_name)
        train_dat = pd.DataFrame(dataset['train'])
        valid_dat = pd.DataFrame(dataset['validation_matched'])
        test_dat = pd.DataFrame(dataset['test_matched'])

        src_list['train_sent1'] = train_dat['premise'].tolist()
        src_list['train_sent2'] = train_dat['hypothesis'].tolist()
        trg_list['train'] = train_dat['label'].tolist()

        src_list['valid_sent1'] = valid_dat['premise'].tolist()
        src_list['valid_sent2'] = valid_dat['hypothesis'].tolist()
        trg_list['valid'] = valid_dat['label'].tolist()

        src_list['test_sent1'] = test_dat['premise'].tolist()
        src_list['test_sent2'] = test_dat['hypothesis'].tolist()
        trg_list['test'] = test_dat['label'].tolist()

    if args.data_name == 'mrpc':
        dataset = load_dataset("glue", args.data_name)
        train_dat = pd.DataFrame(dataset['train'])
        valid_dat = pd.DataFrame(dataset['validation_matched'])
        test_dat = pd.DataFrame(dataset['test_matched'])

        src_list['train_sent1'] = train_dat['sentence1'].tolist()
        src_list['train_sent2'] = train_dat['sentence2'].tolist()
        trg_list['train'] = train_dat['label'].tolist()

        src_list['valid_sent1'] = valid_dat['sentence1'].tolist()
        src_list['valid_sent2'] = valid_dat['sentence2'].tolist()
        trg_list['valid'] = valid_dat['label'].tolist()

        src_list['test_sent1'] = test_dat['sentence1'].tolist()
        src_list['test_sent2'] = test_dat['sentence2'].tolist()
        trg_list['test'] = test_dat['label'].tolist()

    if args.data_name == 'korean_hate_speech':
        args.data_path = os.path.join(args.data_path,'korean-hate-speech-detection')

        train_dat = pd.read_csv(os.path.join(args.data_path, 'train.hate.csv'))
        valid_dat = pd.read_csv(os.path.join(args.data_path, 'dev.hate.csv'))
        test_dat = pd.read_csv(os.path.join(args.data_path, 'test.hate.no_label.csv'))

        train_dat['label'] = train_dat['label'].replace('none', 0)
        train_dat['label'] = train_dat['label'].replace('hate', 1)
        train_dat['label'] = train_dat['label'].replace('offensive', 2)
        valid_dat['label'] = valid_dat['label'].replace('none', 0)
        valid_dat['label'] = valid_dat['label'].replace('hate', 1)
        valid_dat['label'] = valid_dat['label'].replace('offensive', 2)

        src_list['train'] = train_dat['comments'].tolist()
        trg_list['train'] = train_dat['label'].tolist()
        src_list['valid'] = valid_dat['comments'].tolist()
        trg_list['valid'] = valid_dat['label'].tolist()
        src_list['test'] = test_dat['comments'].tolist()
        trg_list['test'] = [0 for _ in range(len(test_dat))]

    return src_list, trg_list

def tokenizing(args, src_list, tokenizer):
    processed_sequences = dict()
    processed_sequences['train'] = dict()
    processed_sequences['valid'] = dict()
    processed_sequences['test'] = dict()

    if args.data_name in []:
        for phase in ['train', 'valid', 'test']:
            encoded_dict = \
            tokenizer(
                src_list[phase],
                max_length=args.src_max_len,
                padding='max_length',
                truncation=True
            )
            processed_sequences[phase]['input_ids'] = encoded_dict['input_ids']
            processed_sequences[phase]['attention_mask'] = encoded_dict['attention_mask']
            if args.model_type == 'bert':
                processed_sequences[phase]['token_type_ids'] = encoded_dict['token_type_ids']

    if args.data_name in []:
        for phase in ['train', 'valid', 'test']:
            encoded_dict = \
            tokenizer(
                src_list['f{phase}_sent1'], src_list['f{phase}_sent2'],
                max_length=args.src_max_len,
                padding='max_length',
                truncation=True
            )
            processed_sequences[phase]['input_ids'] = encoded_dict['input_ids']
            processed_sequences[phase]['attention_mask'] = encoded_dict['attention_mask']
            if args.model_type == 'bert':
                processed_sequences[phase]['token_type_ids'] = encoded_dict['token_type_ids']

    return processed_sequences

def input_to_device(batch_iter, device):

    src_sequence = batch_iter[0]
    src_att = batch_iter[1]
    src_seg = batch_iter[2]
    trg_label = batch_iter[3]

    src_sequence = src_sequence.to(device, non_blocking=True)
    src_att = src_att.to(device, non_blocking=True)
    src_seg = src_seg.to(device, non_blocking=True)
    trg_label = trg_label.to(device, non_blocking=True)

    return src_sequence, src_att, src_seg, trg_label
