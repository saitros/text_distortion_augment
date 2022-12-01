import math
from collections import defaultdict
# Import PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
# Import Huggingface
from transformers import BartModel, BartConfig
#
from ..latent_module.latent import Latent_module 

class custom_Bart(nn.Module):
    def __init__(self, task: str = 'translation', isPreTrain: bool = True, 
                 src_language: str = 'en', trg_language: str = 'en',
                 variational: bool = True, variational_mode_dict: dict = dict(),
                 src_max_len: int = 768, trg_max_len: int = 300,
                 emb_src_trg_weight_sharing: bool = True):
        super().__init__()

        """
        Customized Bart Model
        
        Args:
            encoder_config (dictionary): encoder transformer's configuration
            d_latent (int): latent dimension size
            device (torch.device): 
        Returns:
            log_prob (torch.Tensor): log probability of each word 
            mean (torch.Tensor): mean of latent vector
            log_var (torch.Tensor): log variance of latent vector
            z (torch.Tensor): sampled latent vector
        """
        self.isPreTrain = isPreTrain
        self.variational = variational
        self.src_language = src_language
        self.trg_language = trg_language
        self.emb_src_trg_weight_sharing = emb_src_trg_weight_sharing
        self.model_config = BartConfig.from_pretrained(f'ainize/bart-base-cnn')
        # self.model_config.use_cache = False

        # Token index
        self.pad_idx = self.model_config.pad_token_id
        self.bos_idx = self.model_config.bos_token_id
        self.eos_idx = self.model_config.eos_token_id

        if self.isPreTrain:
            self.model = BartModel.from_pretrained(f'ainize/bart-base-cnn')
        else:
            self.model = BartModel(config=self.model_config)

        self.encoder_model = self.model.get_encoder()
        self.decoder_model = self.model.get_decoder()
        # Shared embedding setting
        self.embeddings = self.model.shared
        # Dimension setting
        self.d_hidden = self.encoder_model.embed_tokens.embedding_dim
        # Language model head setting
        self.lm_head = nn.Linear(self.model_config.d_model, self.model.shared.num_embeddings, bias=False)

        # Variational mode setting
        if variational:
            self.variational_model = variational_mode_dict['variational_model']
            self.variational_token_processing = variational_mode_dict['variational_token_processing']
            self.variational_with_target = variational_mode_dict['variational_with_target']
            self.cnn_encoder = variational_mode_dict['cnn_encoder']
            self.cnn_decoder = variational_mode_dict['cnn_decoder']
            self.latent_add_encoder_out = variational_mode_dict['latent_add_encoder_out']
            self.z_var = variational_mode_dict['z_var']
            self.d_latent = variational_mode_dict['d_latent']

            self.latent_module = Latent_module(d_model=self.d_hidden, d_latent=self.d_latent, 
                                               variational_model=self.variational_model, 
                                               variational_token_processing=self.variational_token_processing,
                                               variational_with_target=self.variational_with_target,
                                               cnn_encoder=self.cnn_encoder, cnn_decoder=self.cnn_decoder,
                                               latent_add_encoder_out=self.latent_add_encoder_out,
                                               z_var=self.z_var, src_max_len=src_max_len, trg_max_len=trg_max_len)

    def forward(self, src_input_ids, src_attention_mask, src_img,
                trg_label, trg_input_ids, trg_attention_mask,
                non_pad_position=None, tgt_subsqeunt_mask=None):

        # Pre_setting for variational model and translation task
        trg_input_ids_copy = trg_input_ids.clone().detach()
        trg_attention_mask_copy = trg_attention_mask.clone().detach()
        trg_input_ids = trg_input_ids[:, :-1]
        trg_attention_mask = trg_attention_mask[:, :-1]

        # Input and output embedding sharing mode
        if self.emb_src_trg_weight_sharing:
            src_input_embeds = self.embeddings(src_input_ids)
            trg_input_embeds = self.embeddings(trg_input_ids)
            trg_input_embeds_ = self.embeddings(trg_input_ids_copy)

        # Encoder Forward
        if self.emb_src_trg_weight_sharing:
            src_encoder_out = self.encoder_model(inputs_embeds=src_input_embeds,
                                                 attention_mask=src_attention_mask)
            src_encoder_out = src_encoder_out['last_hidden_state']
        else:
            src_encoder_out = self.encoder_model(input_ids=src_input_ids,
                                                 attention_mask=src_attention_mask)
            src_encoder_out = src_encoder_out['last_hidden_state']

        # Variational
        if self.variational:
            
            # Tensor dimension transpose
            src_encoder_out = src_encoder_out.transpose(0,1) # [seq_len, batch, d_model]

            # Target sentence latent mapping
            if self.variational_with_target:
                with torch.no_grad():
                    if self.emb_src_trg_weight_sharing:
                        trg_encoder_out = self.encoder_model(inputs_embeds=trg_input_embeds_,
                                                             attention_mask=trg_attention_mask_copy)
                        trg_encoder_out = trg_encoder_out['last_hidden_state']
                    else:
                        trg_encoder_out = self.encoder_model(input_ids=trg_input_ids_copy,
                                                             attention_mask=trg_attention_mask_copy)
                        trg_encoder_out = trg_encoder_out['last_hidden_state']

                trg_encoder_out = trg_encoder_out.transpose(0,1) # [seq_len, batch, d_model]
            else:
                trg_encoder_out = None

            src_encoder_out, dist_loss = self.latent_module(encoder_out_src=src_encoder_out, 
                                                            encoder_out_trg=trg_encoder_out)
            src_encoder_out = src_encoder_out.transpose(0,1)
        else:
            dist_loss = torch.tensor(0, dtype=torch.float)

        # Decoder
        if self.emb_src_trg_weight_sharing:
            model_out = self.decoder_model(inputs_embeds = trg_input_embeds, 
                                           encoder_hidden_states = src_encoder_out,
                                           encoder_attention_mask = src_attention_mask)
            model_out = self.lm_head(model_out['last_hidden_state'])
        else:
            model_out = self.decoder_model(input_ids = trg_input_ids, 
                                           encoder_hidden_states = src_encoder_out,
                                           encoder_attention_mask = src_attention_mask)
            model_out = self.lm_head(model_out['last_hidden_state'])

        if non_pad_position is not None:
            model_out = model_out[non_pad_position]

        return model_out, dist_loss

    def generate(self, src_input_ids, src_attention_mask, beam_size: int = 5, beam_alpha: float = 0.7, repetition_penalty: float = 0.7):

        # Pre_setting
        device = src_input_ids.device
        batch_size = src_input_ids.size(0)
        src_seq_size = src_input_ids.size(1)
        every_batch = torch.arange(0, beam_size * batch_size, beam_size, device=device)

        src_encoder_out = self.encoder_model(input_ids=src_input_ids,
                                             attention_mask=src_attention_mask)
        src_encoder_out = src_encoder_out['last_hidden_state']

        src_encoder_out = src_encoder_out.transpose(0,1)

        if self.variational:
            # Tensor dimension transpose
            src_encoder_out = src_encoder_out.transpose(0,1) # [seq_len, batch, d_model]

            src_encoder_out = self.latent_module.generate(src_encoder_out)

        # Duplicate
        src_encoder_out = src_encoder_out.view(-1, batch_size, 1, self.d_hidden)
        src_encoder_out = src_encoder_out.repeat(1, 1, beam_size, 1)
        src_encoder_out = src_encoder_out.view(src_seq_size, -1, self.d_hidden)

        src_attention_mask = src_attention_mask.view(batch_size, 1, -1)
        src_attention_mask = src_attention_mask.repeat(1, beam_size, 1)
        src_attention_mask = src_attention_mask.view(-1, src_seq_size)

        # Decoding start token setting
        seqs = torch.tensor([[self.bos_idx]], dtype=torch.long, device=device) 
        seqs = seqs.repeat(beam_size * batch_size, 1).contiguous() # (batch_size * k, 1)

        # Scores save vector & decoding list setting
        scores_save = torch.zeros(beam_size * batch_size, 1).to(device) # (batch_size * k, 1)
        top_k_scores = torch.zeros(beam_size * batch_size, 1).to(device) # (batch_size * k, 1)
        complete_seqs = defaultdict(list)
        complete_ind = set()

        for step in range(300):
            model_out = self.decoder_model(input_ids = seqs, 
                                           encoder_hidden_states = src_encoder_out,
                                           encoder_attention_mask = src_attention_mask)
            model_out = self.lm_head(model_out['last_hidden_state'])

            scores = model_out[:,-1,] # Last token

            # Repetition Penalty
            if step >= 1 and repetition_penalty != 0:
                next_ix = next_word_inds.view(-1)
                for ix_ in range(len(next_ix)):
                    if scores[ix_][next_ix[ix_]] < 0:
                        scores[ix_][next_ix[ix_]] *= repetition_penalty
                    else:
                        scores[ix_][next_ix[ix_]] /= repetition_penalty

            # Add score
            scores = top_k_scores.expand_as(scores) + scores

            if step == 0:
                scores = scores[::beam_size] # (batch_size, vocab_num)
                scores[:, self.eos_idx] = float('-inf') # set eos token probability zero in first step
                top_k_scores, top_k_words = scores.topk(beam_size, 1, True, True)  # (batch_size, k) , (batch_size, k)
            else:
                top_k_scores, top_k_words = scores.view(batch_size, -1).topk(beam_size, 1, True, True)

            # Previous and Next word extract
            prev_word_inds = top_k_words // self.model_config.vocab_size # (batch_size * k, out_seq)
            next_word_inds = top_k_words % self.model_config.vocab_size # (batch_size * k, out_seq)
            top_k_scores = top_k_scores.view(batch_size * beam_size, -1) # (batch_size * k, out_seq)
            top_k_words = top_k_words.view(batch_size * beam_size, -1) # (batch_size * k, out_seq)
            seqs = seqs[prev_word_inds.view(-1) + every_batch.unsqueeze(1).repeat(1, beam_size).view(-1)] # (batch_size * k, out_seq)
            seqs = torch.cat([seqs, next_word_inds.view(beam_size * batch_size, -1)], dim=1) # (batch_size * k, out_seq + 1)

            # Find and Save Complete Sequences Score
            if self.eos_idx in next_word_inds:
                eos_ind = torch.where(next_word_inds.view(-1) == self.eos_idx)
                eos_ind = eos_ind[0].tolist()
                complete_ind_add = set(eos_ind) - complete_ind
                complete_ind_add = list(complete_ind_add)
                complete_ind.update(eos_ind)
                if len(complete_ind_add) > 0:
                    scores_save[complete_ind_add] = top_k_scores[complete_ind_add]
                    for ix in complete_ind_add:
                        complete_seqs[ix] = seqs[ix].tolist()

        # If eos token doesn't exist in sequence
        if 0 in scores_save:
            score_save_pos = torch.where(scores_save == 0)
            for ix in score_save_pos[0].tolist():
                complete_seqs[ix] = seqs[ix].tolist()
            scores_save[score_save_pos] = top_k_scores[score_save_pos]

        # Beam Length Normalization
        lp = torch.tensor([len(complete_seqs[i]) for i in range(batch_size * beam_size)], device=device)
        lp = (((lp + beam_size) ** beam_alpha) / ((beam_size + 1) ** beam_alpha)).unsqueeze(1)
        scores_save = scores_save / lp

        # Predicted and Label processing
        _, ind = scores_save.view(batch_size, beam_size, -1).max(1)
        predicted = ind.view(-1) + every_batch
        
        return predicted

    @staticmethod
    def generate_square_subsequent_mask(sz, device):
        mask = torch.tril(torch.ones(sz, sz, dtype=torch.float, device=device))
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)
        return mask