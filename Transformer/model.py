import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerAutoencoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, nhead=8, num_encoder_layers=3, num_decoder_layers=3, bottleneck_dim=6, max_len=50):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len

        self.embed_enc = nn.Embedding(vocab_size, embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim, max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=256)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.bottleneck = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim)
        )
        self.bottleneck_to_emb = nn.Linear(bottleneck_dim, embed_dim)

        self.embed_dec = nn.Embedding(vocab_size, embed_dim)
        self.pos_dec = PositionalEncoding(embed_dim, max_len)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=256)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.output_layer = nn.Linear(embed_dim, vocab_size)

    def encode(self, src):
        emb = self.embed_enc(src)           # [B, L, E]
        emb = self.pos_enc(emb)             # [B, L, E]
        emb = emb.transpose(0, 1)           # [L, B, E]
        enc_out = self.encoder(emb)         # [L, B, E]
        enc_out = enc_out.transpose(0, 1)   # [B, L, E]
        pooled = enc_out.mean(dim=1)        # [B, E]
        bottleneck_vec = self.bottleneck(pooled)  # [B, 6]
        return bottleneck_vec

    def decode(self, bottleneck_vec, tgt):
        emb_tgt = self.embed_dec(tgt)             # [B, L, E]
        emb_tgt = self.pos_dec(emb_tgt)           # [B, L, E]
        emb_tgt = emb_tgt.transpose(0, 1)         # [L, B, E]

        memory = self.bottleneck_to_emb(bottleneck_vec).unsqueeze(0)  # [1, B, E]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)  # [L, L]

        output = self.decoder(tgt=emb_tgt, memory=memory, tgt_mask=tgt_mask)  # [L, B, E]
        output = output.transpose(0, 1)             # [B, L, E]
        return self.output_layer(output)            # [B, L, vocab_size]

    def forward(self, src, tgt):
        bottleneck_vec = self.encode(src)
        return self.decode(bottleneck_vec, tgt)

    @torch.no_grad()
    def generate(self, src, sos_idx, eos_idx, max_len=50):
        self.eval()
        bottleneck_vec = self.encode(src)  # [B, 6]
        batch_size = src.size(0)

        generated = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=src.device)  # [B, 1]

        for _ in range(max_len - 1):
            logits = self.decode(bottleneck_vec, generated)  # [B, L, V]
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [B, 1]
            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == eos_idx).all():
                break

        return generated
