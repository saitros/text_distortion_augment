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
from model.model import TransformerModel
from model.dataset import CustomDataset
from utils import TqdmLoggingHandler, write_log
from task.utils import input_to_device, tokenizing

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

    save_path = os.path.join(args.preprocess_path, args.data_name, args.encoder_model_type)

    with h5py.File(os.path.join(save_path, f'src_len_{args.src_max_len}_processed.hdf5'), 'r') as f:
        train_src_input_ids = f.get('train_src_input_ids')[:]
        train_src_attention_mask = f.get('train_src_attention_mask')[:]
        train_trg_list = f.get('train_label')[:]
        train_trg_list = F.one_hot(torch.tensor(train_trg_list, dtype=torch.long)).numpy()
        if args.encoder_model_type == 'bert':
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
                               trg_list=train_trg_list, src_max_len=args.src_max_len)
    }
    dataloader_dict = {
        'train': DataLoader(dataset_dict['train'], drop_last=True,
                            batch_size=args.batch_size, shuffle=True,
                            pin_memory=True, num_workers=args.num_workers)
    }
    write_log(logger, f"Total number of trainingsets  iterations - {len(dataset_dict['train'])}, {len(dataloader_dict['train'])}")

    # 3) Model loading
    cudnn.benchmark = True
    cls_criterion = nn.CrossEntropyLoss().to(device)
    save_file_name = os.path.join(args.model_save_path, args.data_name, args.encoder_model_type, 'checkpoint_test2.pth.tar')
    checkpoint = torch.load(save_file_name)
    model.load_state_dict(checkpoint['model'])
    write_log(logger, f'Loaded augmenter model from {save_file_name}')

    #===================================#
    #=========Model Train Start=========#
    #===================================#

    start_time_e = time()
    write_log(logger, 'Augmenting start...')
    model.eval()

    aug_sent_dict = {
        'origin': list(),
        'recon': list(),
        'fgsm': list()
    }
    aug_prob_dict = {
        'origin': list(),
        'recon': list(),
        'fgsm': list()
    }

    for i, batch_iter in enumerate(tqdm(dataloader_dict['train'], bar_format='{l_bar}{bar:30}{r_bar}{bar:-2b}')):
        # Input setting
        b_iter = input_to_device(batch_iter, device=device)
        src_sequence, src_att, src_seg, trg_label = b_iter

        # Target Label Setting
        if args.label_flipping:
            fliped_trg_label = torch.flip(trg_label, dims=[1]).to(device)
        else:
            fliped_trg_label = torch.ones_like(trg_label) * 0.5

        # Source De-tokenizing for original sentence
        origin_source = model.tokenizer.batch_decode(src_sequence, skip_special_tokens=True)
        aug_sent_dict['origin'].extend(origin_source)
        aug_prob_dict['origin'].extend(trg_label.to('cpu').numpy().tolist())

        # Generate sentence
        with torch.no_grad():

            # Encoding
            encoder_out = model.encode(input_ids=src_sequence, attention_mask=src_att)
            hidden_states = encoder_out
            if args.classify_method == 'latent_out':
                latent_out = model.latent_encode(encoder_out=encoder_out)
                hidden_states = latent_out

            # Reconstruction
            latent_out = None
            if args.encoder_out_mix_ratio != 0:
                latent_out = model.latent_encode(encoder_out=encoder_out)

            if args.test_decoding_strategy == 'greedy':
                recon_out = model(input_ids=src_sequence, attention_mask=src_att, encoder_out=encoder_out, latent_out=latent_out)
            elif args.test_decoding_strategy == 'beam':
                recon_out = model.generate(encoder_out=encoder_out, latent_out=latent_out, attention_mask=src_att, beam_size=args.beam_size,
                                           beam_alpha=args.beam_alpha, repetition_penalty=args.repetition_penalty, device=device)
            elif args.test_decoding_strategy in ['multinomial', 'topk', 'topp']:
                recon_out = model.generate_sample(encoder_out=encoder_out, latent_out=latent_out, attention_mask=src_att,
                                                  sampling_strategy=args.test_decoding_strategy, device=device,
                                                  topk=args.topk, topp=args.topp, softmax_temp=args.multinomial_temperature)
            decoded_output = model.tokenizer.batch_decode(recon_out, skip_special_tokens=True)

        # Get classification probability for decoded_output
        classifier_out = model.classify(hidden_states=hidden_states)

        aug_sent_dict['recon'].extend(decoded_output)
        aug_prob_dict['recon'].extend(F.softmax(classifier_out, dim=1).detach().cpu().numpy())

        # FGSM - take gradient of classifier output w.r.t. encoder output
        hidden_states_grad_true = hidden_states.clone().detach().requires_grad_(True)
        classifier_out = model.classify(hidden_states=hidden_states_grad_true)

        for _ in range(3):

            model.zero_grad()
            cls_loss = cls_criterion(classifier_out, fliped_trg_label)
            cls_loss.backward()
            hidden_states_grad = hidden_states_grad_true.grad.data.sign()
            
            # Gradient-based Modification
            if args.classify_method == 'latent_out':
                encoder_out_copy = encoder_out.clone().detach()
                latent_out_copy = hidden_states_grad_true - (args.grad_epsilon * hidden_states_grad)
                hidden_states_grad_true = latent_out_copy.clone().detach().requires_grad_(True)
            else:
                encoder_out_copy = hidden_states_grad_true - (args.grad_epsilon * hidden_states_grad)
                latent_out_copy = None
                hidden_states_grad_true = encoder_out_copy.clone().detach().requires_grad_(True)

            classifier_out = model.classify(hidden_states=hidden_states_grad_true)

        # FGSM - generate sentence
        with torch.no_grad():
            if args.test_decoding_strategy == 'beam':
                recon_out = model.generate(encoder_out=encoder_out_copy, latent_out=latent_out_copy, attention_mask=src_att, beam_size=args.beam_size,
                                           beam_alpha=args.beam_alpha, repetition_penalty=args.repetition_penalty, device=device)
            elif args.test_decoding_strategy in ['greedy', 'multinomial', 'topk', 'topp']:
                recon_out = model.generate_sample(encoder_out=encoder_out, latent_out=latent_out, attention_mask=src_att,
                                                  sampling_strategy=args.test_decoding_strategy, device=device,
                                                  topk=args.topk, topp=args.topp, softmax_temp=args.multinomial_temperature)
            decoded_output = model.tokenizer.batch_decode(recon_out, skip_special_tokens=True)

            # Get classification probability for decoded_output
            classifier_out_fgsm = model.classify(hidden_states=encoder_out_copy)
        aug_sent_dict['fgsm'].extend(decoded_output)
        aug_prob_dict['fgsm'].extend(F.softmax(classifier_out_fgsm, dim=1).to('cpu').numpy().tolist())

        if args.debuging_mode:
            print(aug_prob_dict)
            break

    # POST-PROCESSING
    processed_aug_seq = {}
    processed_aug_seq['origin'] = {}
    processed_aug_seq['recon'] = {}
    processed_aug_seq['fgsm'] = {}

    for phase in ['origin', 'recon', 'fgsm']:
        encoded_dict = \
            model.tokenizer(aug_sent_dict[phase],
                            max_length=args.src_max_len,
                            padding='max_length',
                            truncation=True
        )
        processed_aug_seq[phase]['input_ids'] = encoded_dict['input_ids']
        processed_aug_seq[phase]['attention_mask'] = encoded_dict['attention_mask']
        if args.encoder_model_type == 'bert':
            processed_aug_seq[phase]['token_type_ids'] = encoded_dict['token_type_ids']

        if args.augmenting_label == 'soft':
            processed_aug_seq[phase]['label'] = aug_prob_dict[phase]
        elif args.augmenting_label == 'hard':
            processed_aug_seq[phase]['label'] = np.argmax(np.array(aug_prob_dict[phase]), axis=1) # Need Check
        else:
            raise ValueError

    # Save with h5py
    save_path = os.path.join(args.preprocess_path, args.data_name, args.encoder_model_type)

    with h5py.File(os.path.join(save_path, 'processed_aug.hdf5'), 'w') as f:
        # origin data
        f.create_dataset('train_src_input_ids', data=processed_aug_seq['origin']['input_ids'])
        f.create_dataset('train_src_attention_mask', data=processed_aug_seq['origin']['attention_mask'])
        f.create_dataset('train_label', data=processed_aug_seq['origin']['label'])
        # recon data
        f.create_dataset('train_recon_src_input_ids', data=processed_aug_seq['recon']['input_ids'])
        f.create_dataset('train_recon_src_attention_mask', data=processed_aug_seq['recon']['attention_mask'])
        f.create_dataset('train_recon_label', data=processed_aug_seq['recon']['label'])
        # fgsm data
        f.create_dataset('train_fgsm_src_input_ids', data=processed_aug_seq['fgsm']['input_ids'])
        f.create_dataset('train_fgsm_src_attention_mask', data=processed_aug_seq['fgsm']['attention_mask'])
        f.create_dataset('train_fgsm_label', data=processed_aug_seq['fgsm']['label'])
        if args.encoder_model_type == 'bert':
            f.create_dataset('train_src_token_type_ids', data=processed_aug_seq['origin']['token_type_ids'])
            f.create_dataset('train_recon_src_token_type_ids', data=processed_aug_seq['recon']['token_type_ids'])
            f.create_dataset('train_fgsm_src_token_type_ids', data=processed_aug_seq['fgsm']['token_type_ids'])

    print('Postprocessing Done!')
