# Import PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.cuda.amp import autocast
from torch.nn import functional as F
# Import Huggingface
from transformers import PretrainedConfig, AutoModel, AutoTokenizer
from transformers import AlbertConfig, AlbertForSequenceClassification
from transformers import DebertaConfig, DebertaForSequenceClassification
from transformers import BertConfig, BertModel, BertForSequenceClassification
# Import Custom Modules
from model.utils import return_model_name

class TransformerModel(nn.Module):
    def __init__(self, model_type: str = 'bert', num_labels: int = 2,
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
        self.model_type = model_type

        # Token index & dimension
        model_name = return_model_name(self.model_type)
        self.model_config = PretrainedConfig.from_pretrained(model_name)
        self.pad_idx = self.model_config.pad_token_id
        self.bos_idx = self.model_config.bos_token_id
        self.eos_idx = self.model_config.eos_token_id
        self.num_labels = num_labels

        self.d_hidden = self.model_config.hidden_size
        self.d_embedding = int(self.model_config.hidden_size / 2)
        self.vocab_num = self.model_config.vocab_size

        # Pre-trained Model Setting
        self.encoder_model = AutoModel.from_pretrained(model_name)

        # Dimension Reduction
        self.linear_encoding = nn.Linear(self.d_hidden, self.d_embedding)
        self.linear_decoding = nn.Linear(self.d_embedding, self.d_hidden)

        # Linear Model Setting
        self.decoder_linear = nn.Linear(self.d_hidden, self.d_embedding)
        self.decoder_norm = nn.LayerNorm(self.d_embedding, eps=1e-12)
        self.decoder_augmenter = nn.Linear(self.d_embedding, self.vocab_num)

        # Tokenizer Setting
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    @autocast()
    def forward(self, src_input_ids, src_attention_mask, src_token_type_ids):

        # Encoding
        encoder_out = self.encoder_model(input_ids=src_input_ids, 
                                         attention_mask=src_attention_mask,
                                         token_type_ids=src_token_type_ids)
        encoder_out = encoder_out['last_hidden_state']

        # Linear encoding (Motivated by WAE)
        z = self.linear_encoding(encoder_out)
        decoder_in = self.linear_decoding(z)

        # Decoding
        decoder_out = self.dropout(F.gelu(self.decoder_linear(decoder_in)))
        decoder_out = self.decoder_augmenter(self.decoder_norm(decoder_out))

        return decoder_out, z