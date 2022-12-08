# Import PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.cuda.amp import autocast
from torch.nn import functional as F
# Import Huggingface
from transformers import BertConfig, BertModel, BertForSequenceClassification
from transformers import DebertaConfig, DebertaForSequenceClassification
from transformers import AlbertConfig, AlbertForSequenceClassification

class AugModel(nn.Module):
    def __init__(self, encoder_model_type: str = 'bert', decoder_model_type: str = 'bert',
                 isPreTrain: bool = True, z_variation: float = 2, 
                 dropout: float = 0.3):
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

        # Token index & dimension
        self.model_config = BertConfig.from_pretrained('bert-base-cased')
        self.pad_idx = self.model_config.pad_token_id
        self.bos_idx = self.model_config.bos_token_id
        self.eos_idx = self.model_config.eos_token_id

        self.d_hidden = self.model_config.hidden_size
        self.d_embedding = int(self.model_config.hidden_size / 2)
        self.vocab_num = self.model_config.vocab_size

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

            self.decoder_linear1 = nn.Linear(self.d_hidden, self.d_embedding)
            self.decoder_norm = nn.LayerNorm(self.d_embedding, eps=1e-12)
            self.decoder_linear2 = nn.Linear(self.d_embedding, self.vocab_num)
        else:
            raise Exception('''It's not ready...''')

    @autocast()
    def forward(self, src_input_ids, src_attention_mask, src_token_type_ids):

        # Encoding
        encoder_out = self.encoder_model(input_ids=src_input_ids, 
                                         attention_mask=src_attention_mask,
                                         token_type_ids=src_token_type_ids)
        encoder_out = encoder_out['last_hidden_state']

        # Sampling Z
        z = self.z_variation * torch.rand_like(encoder_out)

        # Decoding
        decoder_out = self.decoder_model(inputs_embeds=encoder_out + z, 
                                         attention_mask=src_attention_mask,
                                         token_type_ids=src_token_type_ids)
        decoder_out = decoder_out['last_hidden_state']
        
        decoder_out = self.dropout(F.gelu(self.decoder_linear1(decoder_out)))
        decoder_out = self.decoder_linear2(self.decoder_norm(decoder_out))

        return encoder_out, decoder_out, z

    @autocast()
    def generate(self, src_input_ids, src_attention_mask, src_token_type_ids):
        # Encoding
        encoder_out = self.encoder_model(input_ids=src_input_ids, 
                                         attention_mask=src_attention_mask,
                                         token_type_ids=src_token_type_ids)
        encoder_out = encoder_out['last_hidden_state']

        z = (self.z_variation*2) * torch.rand_like(encoder_out)

        # Decoding
        decoder_out = self.decoder_model(inputs_embeds=encoder_out + z, 
                                         attention_mask=src_attention_mask,
                                         token_type_ids=src_token_type_ids)
        decoder_out = decoder_out['last_hidden_state']
        
        decoder_out = self.dropout(F.gelu(self.decoder_linear1(decoder_out)))
        decoder_out = self.decoder_linear2(self.decoder_norm(decoder_out))

        return decoder_out

class ClsModel(nn.Module):
    def __init__(self, model_type: str = 'bert', num_labels: int = 2):
        super().__init__()

        self.model_type = model_type
        self.num_labels = num_labels

        # Token index & dimension
        if self.model_type == 'bert':
            self.model_config = BertConfig.from_pretrained('bert-base-cased')
            self.cls_model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=self.num_labels)
        if self.model_type == 'deberta':
            self.model_config = DebertaConfig.from_pretrained('hf-internal-testing/tiny-random-deberta')
            self.cls_model = DebertaForSequenceClassification.from_pretrained("hf-internal-testing/tiny-random-deberta", num_labels=self.num_labels)
        if self.model_type == 'albert':
            self.model_config = AlbertConfig.from_pretrained('textattack/albert-base-v2-imdb')
            self.cls_model = AlbertForSequenceClassification.from_pretrained("textattack/albert-base-v2-imdb", num_labels=self.num_labels)

    @autocast()
    def forward(self, src_input_ids, src_attention_mask, src_token_type_ids):
        if self.model_type == 'bert':
            model_out = self.cls_model(input_ids=src_input_ids, 
                                       attention_mask=src_attention_mask,
                                       token_type_ids=src_token_type_ids)
        else:
            model_out = self.cls_model(input_ids=src_input_ids, 
                                       attention_mask=src_attention_mask)
        model_out = model_out['logits']

        return model_out