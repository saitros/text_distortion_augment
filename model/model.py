# Import PyTorch
import torch
import torch.nn as nn
from torch.nn import functional as F
# Import Huggingface
from transformers import BertConfig, BertModel

class AugModel(nn.Module):
    def __init__(self, encoder_model_type: str = 'bert', decoder_model_type: str = 'bert',
                 isPreTrain: bool = True, z_variation: float = 2, 
                 src_max_len: int = 768, dropout: float = 0.3):
        super().__init__()

        """
        Initialize WAE model
        
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
        self.z_variation = z_variation
        self.dropout = nn.Dropout(dropout)
        self.encoder_model_type = encoder_model_type
        self.decoder_model_type = decoder_model_type

        # Token index
        self.model_config = BertConfig.from_pretrained('bert-base-cased')
        self.pad_idx = self.model_config.pad_token_id
        self.bos_idx = self.model_config.bos_token_id
        self.eos_idx = self.model_config.eos_token_id

        # Encoder Model Setting
        if self.encoder_model_type == 'bert':
            if self.isPreTrain:
                self.encoder_model = BertModel.from_pretrained('bert-base-cased')
            else:
                self.encoder_model = BertModel(config=self.model_config)
        else:
            raise Exception('''It's not ready...''')

        # Decoder Model Setting
        if self.decoder_model_type == 'bert':
            if self.isPreTrain:
                self.decoder_model = BertModel.from_pretrained('bert-base-cased')
            else:
                self.decoder_model = BertModel(config=self.model_config)

            self.decoder_linear1 = nn.Linear(self.model_config.hidden_size, 256)
            self.decoder_norm = nn.LayerNorm(256, eps=1e-12)
            self.decoder_linear2 = nn.Linear(256, self.model_config.vocab_size)
        else:
            raise Exception('''It's not ready...''')

    def forward(self, src_input_ids, src_attention_mask, src_token_type_ids, non_pad_position = None):

        # Encoding
        encoder_out = self.encoder_model(input_ids=src_input_ids, 
                                         attention_mask=src_attention_mask,
                                         token_type_ids=src_token_type_ids)
        encoder_out = encoder_out['last_hidden_state']
        # encoder_out = encoder_out['last_hidden_state'].mean(dim=1)

        # Sampling Z
        z = self.z_variation * encoder_out.data.new(encoder_out.size()).normal_()

        # Decoding
        decoder_out = self.decoder_model(inputs_embeds=encoder_out, 
                                         attention_mask=src_attention_mask,
                                         token_type_ids=src_token_type_ids)
        decoder_out = decoder_out['last_hidden_state']
        
        if non_pad_position is not None:
            decoder_out = decoder_out[non_pad_position]
        decoder_out = self.dropout(F.gelu(self.decoder_linear1(decoder_out)))
        decoder_out = self.decoder_linear2(self.decoder_norm(decoder_out))

        return encoder_out, decoder_out, z

    def generate(self, src_input_ids, src_attention_mask, src_token_type_ids):
        # Encoding
        encoder_out = self.encoder_model(input_ids=src_input_ids, attention_mask=src_attention_mask,
                                 token_type_ids=src_token_type_ids)
        encoder_out = encoder_out['last_hidden_state'].mean(dim=1)

        # Sampling Z
        z = self.sample_z(encoder_out, sigma=self.z_variation * 2)

        # Decoding
        decoder_out = self.dropout(F.gelu(self.decoder_linear1(encoder_out)))
        decoder_out = self.decoder_linear2(self.decoder_norm(decoder_out))

        return decoder_out