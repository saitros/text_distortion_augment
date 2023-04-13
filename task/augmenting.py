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
from utils import TqdmLoggingHandler, write_log, return_model_name
from task.utils import input_to_device, tokenizing, data_load, data_sampling

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
    #=============Data Load=============#
    #===================================#

    write_log(logger, "Load data...")

    start_time = time()
    total_src_list, total_trg_list = data_load(args)
    total_src_list, total_trg_list = data_sampling(args, total_src_list, total_trg_list)
    num_labels = len(set(total_trg_list['train']))
    write_log(logger, 'Data loading done!')

    # 2) Dataloader setting
    write_log(logger, "CustomDataset setting...")
    tokenizer_name = return_model_name(args.encoder_model_type)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    src_vocab_num = tokenizer.vocab_size

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
        'train': CustomDataset(tokenizer=tokenizer,
                               src_list=total_src_list['train'], src_list2=total_src_list['train2'], 
                               trg_list=total_trg_list['train'], src_max_len=args.src_max_len),
    }
    dataloader_dict = {
        'train': DataLoader(dataset_dict['train'], drop_last=True,
                            batch_size=args.batch_size, shuffle=True,
                            pin_memory=True, num_workers=args.num_workers)
    }
    write_log(logger, f"Total number of trainingsets iterations - {len(dataset_dict['train'])}, {len(dataloader_dict['train'])}")

    # 3) Model loading
    cudnn.benchmark = True
    cls_criterion = nn.CrossEntropyLoss().to(device)
    save_file_name = os.path.join(args.model_save_path, args.data_name, args.encoder_model_type, f'checkpoint_seed_{args.random_seed}.pth.tar')
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
        'augment': list()
    }
    aug_prob_dict = {
        'origin': list(),
        'recon': list(),
        'augment': list()
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
            if model.latent_out_mix_ratio != 0:
                latent_out = model.latent_encode(encoder_out=encoder_out)

            if args.test_decoding_strategy == 'greedy':
                recon_out = model(input_ids=src_sequence, attention_mask=src_att, encoder_out=encoder_out, latent_out=latent_out).argmax(dim=2)
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

        for _ in range(args.epsilon_repeat):

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
            classifier_out_augment = model.classify(hidden_states=encoder_out_copy)
        aug_sent_dict['augment'].extend(decoded_output)
        aug_prob_dict['augment'].extend(F.softmax(classifier_out_augment, dim=1).to('cpu').numpy().tolist())

        if args.debuging_mode:
            print(aug_prob_dict)
            break

    # Save with csv
    save_path = os.path.join(args.preprocess_path, args.data_name, args.encoder_model_type)
    save_dat = pd.DataFrame(
        {
            'origin_sent': aug_sent_dict['origin'],
            'origin_prob': aug_prob_dict['origin'],
            'recon_sent': aug_sent_dict['recon'],
            'recon_prob': aug_prob_dict['recon'],
            'augment_sent': aug_sent_dict['augment'],
            'augment_prob': aug_prob_dict['augment'],
        }
    )
    save_dat.to_csv(os.path.join(save_path, 'aug_dat.csv'), index=False)