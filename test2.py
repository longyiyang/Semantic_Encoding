import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

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
    def __init__(self, vocab_size, embed_dim=64, nhead=8, num_encoder_layers=3, num_decoder_layers=3, bottleneck_dim=6, max_len=20):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_len = max_len

        # 编码器
        self.embed_enc = nn.Embedding(vocab_size, embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim, max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=256)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # 瓶颈
        self.bottleneck = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim)
        )

        # 瓶颈向量映射到解码器memory维度（embedding维度）
        self.bottleneck_to_emb = nn.Linear(bottleneck_dim, embed_dim)

        # 解码器
        self.embed_dec = nn.Embedding(vocab_size, embed_dim)
        self.pos_dec = PositionalEncoding(embed_dim, max_len)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=256)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # 输出层
        self.output_layer = nn.Linear(embed_dim, vocab_size)

    def encode(self, src):
        # src: [B, L]
        emb = self.embed_enc(src)        # [B, L, E]
        emb = self.pos_enc(emb)          # 加位置编码
        emb = emb.transpose(0, 1)        # [L, B, E]
        enc_out = self.encoder(emb)      # [L, B, E]
        enc_out = enc_out.transpose(0,1) # [B, L, E]
        pooled = enc_out.mean(dim=1)     # 全局平均池化 [B, E]
        bottleneck_vec = self.bottleneck(pooled)  # [B, bottleneck_dim]
        return bottleneck_vec

    def decode(self, bottleneck_vec, tgt, teacher_forcing_ratio=0.5):
        """
        tgt: [B, L] 解码目标序列
        """
        batch_size, seq_len = tgt.shape

        # 解码器输入准备
        emb_tgt = self.embed_dec(tgt)      # [B, L, E]
        emb_tgt = self.pos_dec(emb_tgt)    # 加位置编码
        emb_tgt = emb_tgt.transpose(0,1)   # [L, B, E]

        # memory是编码器输出（这里用瓶颈向量扩展成1个时间步）
        memory = bottleneck_vec.unsqueeze(0)  # [1, B, bottleneck_dim]
        memory = torch.relu(self.bottleneck_to_emb(memory))  # [1, B, E]

        # tgt_mask 防止看到未来信息
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(tgt.device)  # [L, L]

        output = self.decoder(tgt=emb_tgt, memory=memory, tgt_mask=tgt_mask)  # [L, B, E]
        output = output.transpose(0,1)  # [B, L, E]
        logits = self.output_layer(output)  # [B, L, vocab_size]
        return logits

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        bottleneck_vec = self.encode(src)
        logits = self.decode(bottleneck_vec, tgt, teacher_forcing_ratio)
        return logits

# ----------------------- 数据准备 ------------------------

char2idx = {'<pad>':0, '挖':1, '掘':2, '机':3}
idx2char = {v:k for k,v in char2idx.items()}

def encode_text(text, max_len=5):
    ids = [char2idx.get(ch, 0) for ch in text]
    if len(ids) < max_len:
        ids += [0]*(max_len - len(ids))
    else:
        ids = ids[:max_len]
    return torch.tensor(ids)

train_data = [encode_text("挖掘机")] * 1000

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# ----------------------- 训练 ------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerAutoencoder(vocab_size=len(char2idx), max_len=5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(ignore_index=0)

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch_x in dataloader:
        batch_x = batch_x.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x, batch_x)  # 输入和目标都用batch_x
        loss = criterion(outputs.view(-1, outputs.size(-1)), batch_x.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

for epoch in range(10):
    loss = train_epoch(model, train_loader, optimizer, criterion, device)
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

# ----------------------- 测试重构 ------------------------

model.eval()
test_seq = encode_text("挖掘机").unsqueeze(0).to(device)
with torch.no_grad():
    logits = model(test_seq, test_seq)  # 用target做输入测试
    preds = logits.argmax(dim=-1).squeeze(0).cpu().tolist()
    pred_chars = [idx2char.get(i, '') for i in preds]
    print("预测重构:", "".join(pred_chars).strip('<pad>'))
