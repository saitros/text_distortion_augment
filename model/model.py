import math
# Import PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.cuda.amp import autocast
from torch.nn import functional as F
# Import Huggingface
from transformers import PretrainedConfig, AutoModel, AutoTokenizer
# Import Custom Modules
from utils import return_model_name
from model.utils import model_setting

class TransformerModel(nn.Module):
    def __init__(self, model_type: str = 'bart', src_max_len: int = 150,
                 isPreTrain: bool = True, num_labels: int = 2, dropout: float = 0.3):
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
        self.src_max_len = src_max_len
        self.dropout = nn.Dropout(dropout)
        self.model_type = model_type

        # Model setting
        model_name = return_model_name(self.model_type)
        encoder, decoder, model_config = model_setting(model_name, self.isPreTrain)

        self.model_config = model_config
        self.d_hidden = model_config.d_model
        self.d_embedding = int(self.d_hidden / 2)
        self.num_labels = num_labels
        self.vocab_num = model_config.vocab_size

        # Encoder setting
        self.encoder = encoder

        # Latent Setting
        self.latent_encoder = nn.Linear(self.d_hidden, self.d_embedding)
        self.latent_decoder = nn.Linear(self.d_embedding, self.d_hidden)

        # Classifier
        self.classifier1 = nn.Linear(self.d_hidden, self.d_embedding)
        self.classifier2 = nn.Linear(self.d_embedding, self.d_embedding)
        self.classifier3 = nn.Linear(self.d_embedding, self.num_labels)
        self.leaky_relu = nn.LeakyReLU(0.1)

        # Augmenter Model Setting
        self.decoder = decoder
        self.decoder_linear = nn.Linear(self.d_hidden, self.d_embedding)
        self.decoder_norm = nn.LayerNorm(self.d_embedding, eps=1e-12)
        self.decoder_augmenter = nn.Linear(self.d_embedding, self.vocab_num)

        # Tokenizer Setting
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pad_idx = self.tokenizer.pad_token_id
        self.decoder_start_token_id = self.model_config.decoder_start_token_id
        if self.model_type == 'bert':
            self.bos_idx = self.tokenizer.cls_token_id
            self.eos_idx = self.tokenizer.sep_token_id
        else:
            self.bos_idx = self.tokenizer.bos_token_id
            self.eos_idx = self.tokenizer.eos_token_id

    def encode(self, input_ids, attention_mask):
        encoder_out = self.encoder(input_ids=input_ids, 
                                   attention_mask=attention_mask)
        encoder_out = encoder_out['last_hidden_state']

        return encoder_out

    def latent_encode(self, encoder_out):
        latent_out = encoder_out.mean(dim=1) # (batch_size, d_hidden)
        latent_encoder_out = self.latent_encoder(latent_out) # (batch_size, d_embedding)
        latent_decoder_out = self.latent_decoder(latent_encoder_out) # (batch_size, d_hidden)

        return latent_decoder_out, latent_encoder_out

    def classify(self, latent_out):

        classifier_out = self.dropout(self.leaky_relu(self.classifier1(latent_out)))
        classifier_out = self.dropout(self.leaky_relu(self.classifier2(classifier_out)))
        classifier_out = self.classifier3(classifier_out)

        return classifier_out

    def forward(self, input_ids, attention_mask, encoder_out, latent_out):

        seq_len = input_ids.size(1)

        decoder_input_ids = shift_tokens_right(
            input_ids, self.pad_idx, self.decoder_start_token_id
        )

        hidden_states = torch.add((0.3 * encoder_out), (0.7 * latent_out.unsqueeze(1))) # 이거 애매
        
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask
        )
        decoder_outputs = decoder_outputs['last_hidden_state']

        decoder_outputs = self.dropout(F.gelu(self.decoder_linear(decoder_outputs)))
        decoder_outputs = self.decoder_augmenter(self.decoder_norm(decoder_outputs))

        return decoder_outputs

    def generate(self, src_input_ids, src_attention_mask, latent_z): # Need to fix

        if self.encoder_out_ratio == 0:
            encoder_attention_mask = src_attention_mask[:,0].unsqueeze(1)
        else:
            encoder_attention_mask = src_attention_mask

        decoder_input_ids = shift_tokens_right(
            src_input_ids, self.pad_idx, self.decoder_start_token_id
        )

        # Decoding
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=latent_z,
            encoder_attention_mask=encoder_attention_mask
        )
        decoder_outputs = decoder_outputs['last_hidden_state']

        # Reconstruct
        recon_out = self.reconstruct(decoder_out=decoder_outputs)

        return recon_out

class ClassifierModel(nn.Module):
    def __init__(self, d_latent, num_labels: int = 2, dropout: float = 0.3):
        super().__init__()

        self.linear1 = nn.Linear(d_latent, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, num_labels)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, hidden_state):
        # encoder_out = encoder_out.mean(dim=1)
        out = self.dropout(self.leaky_relu(self.linear1(hidden_state)))
        out = self.dropout(self.leaky_relu(self.linear2(out)))
        out = self.linear3(out)

        return out

def _prepare_bart_decoder_inputs(
    config, input_ids, decoder_input_ids=None, decoder_padding_mask=None, causal_mask_dtype=torch.float32
):
    """Prepare masks that ignore padding tokens in the decoder and a causal mask for the decoder if
    none are provided. This mimics the default behavior in fairseq. To override it pass in masks.
    Note: this is not called during generation
    """
    pad_token_id = config.pad_token_id
    if decoder_input_ids is None:
        decoder_input_ids = shift_tokens_right(input_ids, pad_token_id)
    bsz, tgt_len = decoder_input_ids.size()
    if decoder_padding_mask is None:
        decoder_padding_mask = make_padding_mask(decoder_input_ids, pad_token_id)
    else:
        decoder_padding_mask = invert_mask(decoder_padding_mask)
    causal_mask = torch.triu(fill_with_neg_inf(torch.zeros(tgt_len, tgt_len)), 1).to(
        dtype=causal_mask_dtype, device=decoder_input_ids.device
    )
    return decoder_input_ids, decoder_padding_mask, causal_mask

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

def make_padding_mask(input_ids, padding_idx=1):
    """True for pad tokens"""
    padding_mask = input_ids.eq(padding_idx)
    if not padding_mask.any():
        padding_mask = None
    return padding_mask

def invert_mask(attention_mask):
    assert attention_mask.dim() == 2
    return attention_mask.eq(0)

def fill_with_neg_inf(t):
    """FP16-compatible function that fills a input_ids with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)

class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        pe = torch.zeros(max_len, d_model, dtype=torch.float)
        pe.require_grad = False

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]