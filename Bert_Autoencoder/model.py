import torch
import torch.nn as nn
from transformers import BertModel

class TextAutoencoder(nn.Module):
    def __init__(self, hidden_dim=6, bert_model='./Bert_Autoencoder/bert/bert-base-chinese'):
        super().__init__()
        self.encoder = BertModel.from_pretrained(bert_model)
        
        self.projection = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim)
        )
        
        self.decoder_input_proj = nn.Linear(hidden_dim, self.encoder.config.hidden_size)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.encoder.config.hidden_size,
            nhead=8,
            dim_feedforward=512
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)
        
        self.output_proj = nn.Linear(self.encoder.config.hidden_size, self.encoder.config.vocab_size)
        self.token_embedding = nn.Embedding(self.encoder.config.vocab_size, self.encoder.config.hidden_size)
    
    def forward(self, input_ids, attention_mask, decoder_input_ids):
        encoder_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = encoder_output.last_hidden_state[:, 0, :]
        latent = self.projection(cls_embedding)
        projected_latent = self.decoder_input_proj(latent).unsqueeze(1)

        tgt_embeddings = self.token_embedding(decoder_input_ids).transpose(0, 1)
        memory = projected_latent.transpose(0, 1)
        output = self.transformer_decoder(tgt_embeddings, memory)
        output = output.transpose(0, 1)
        logits = self.output_proj(output)
        return logits
