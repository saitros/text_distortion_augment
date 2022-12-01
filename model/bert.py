# Import PyTorch
import torch
import torch.nn as nn
# Import Huggingface
# T5
from transformers import BertConfig, BertModel
from transformers import ViTFeatureExtractor, ViTModel
from ..latent_module.latent import Latent_module 

class custom_Bert(nn.Module):
    def __init__(self, num_class: int = None, isPreTrain: bool = True, 
                 variational: bool = True, variational_mode_dict: dict = dict(),
                 src_max_len: int = 768):
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

        # Token index
        self.model_config = BertConfig.from_pretrained('bert-base-cased')
        self.pad_idx = self.model_config.pad_token_id
        self.bos_idx = self.model_config.bos_token_id
        self.eos_idx = self.model_config.eos_token_id

        if self.isPreTrain:
            self.model = BertModel.from_pretrained('bert-base-cased')
        else:
            self.model = BertModel(config=self.model_config)

        self.embedding = self.model.embeddings
        self.pooler = self.model.pooler
        self.prediction_head = nn.Linear(768, num_class)

    def forward(self, src_input_ids, src_attention_mask, 
                src_token_type_ids, trg_label)

        # Embedding
        encoder_out = self.embedding(src_input_ids)

        # Attention mask processing
        if self.task == 'multi-modal_classification':
            img_attention_mask = torch.ones(src_input_ids.size(0), 197, dtype=torch.long, device=src_attention_mask.device) # vit-226's patch length is 197
            src_attention_mask = torch.cat((img_attention_mask, src_attention_mask), axis=1)
            new_attention_mask = self.txt_model.get_extended_attention_mask(src_attention_mask, 
                                                                            src_attention_mask.shape, src_attention_mask.device)
        else:
            new_attention_mask = self.txt_model.get_extended_attention_mask(src_attention_mask, 
                                                                            src_attention_mask.shape, src_attention_mask.device)

        encoder_out = self.encoder(hidden_states=encoder_out, 
                                   attention_mask=new_attention_mask)[0]

        # Latent
        if self.variational:
            # Target sentence latent mapping
            with torch.no_grad():
                encoder_out_trg = self.encoder1_embedding(trg_input_ids)
                new_attention_mask2 = self.model1.get_extended_attention_mask(trg_attention_mask, 
                                                                              trg_attention_mask.shape, self.device)
                for i in range(len(self.encoder1_model)):
                    encoder_out_trg, _ = self.encoder1_model[i](hidden_states=encoder_out_trg, 
                                                                attention_mask=new_attention_mask2)

            encoder_out, dist_loss = self.latent_module(encoder_out, encoder_out_trg)
        else:
            dist_loss = 0

        # Encoder2 Forward
        encoder_out = self.pooler(encoder_out)
        # encoder_out = encoder_out[:,-1,:]
        model_out = self.prediction_head(encoder_out)

        return model_out, dist_loss