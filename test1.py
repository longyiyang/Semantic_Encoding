import torch
import torch.nn as nn
import torch.nn.functional as F

class CharAutoencoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, enc_hidden=128, dec_hidden=128, bottleneck_dim=6):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        
        # 编码器：2层双向LSTM
        self.encoder = nn.LSTM(embed_dim, enc_hidden, num_layers=2, batch_first=True, bidirectional=True)
        
        # 非线性瓶颈降维
        self.bottleneck = nn.Sequential(
            nn.Linear(enc_hidden*2, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim)
        )
        
        # 解码器输入由瓶颈映射到隐藏状态初始值
        self.bottleneck_to_dec = nn.Linear(bottleneck_dim, dec_hidden)
        
        # 解码器LSTM
        self.decoder = nn.LSTM(embed_dim, dec_hidden, batch_first=True)
        
        # 输出层预测字符
        self.output_layer = nn.Linear(dec_hidden, vocab_size)
    
    def encode(self, x):
        emb = self.embed(x)  # [B, L, E]
        enc_out, _ = self.encoder(emb)  # [B, L, 2*H]
        pooled = enc_out.mean(dim=1)    # [B, 2*H]
        bottleneck_vec = self.bottleneck(pooled)  # [B, 6]
        return bottleneck_vec
    
    def decode(self, bottleneck_vec, targets=None, teacher_forcing_ratio=0.5):
        """
        解码器输入用Teacher Forcing：
        targets: [B, L] - ground truth indices for decoder input & loss
        """
        batch_size = bottleneck_vec.size(0)
        seq_len = targets.size(1) if targets is not None else 20  # 预测长度
        
        hidden = self.bottleneck_to_dec(bottleneck_vec).unsqueeze(0)  # (1, B, dec_hidden)
        cell = torch.zeros_like(hidden)
        
        # 解码器第一个输入一般用start token，这里用0假设是<pad>/<start>
        input_token = torch.zeros(batch_size, 1, dtype=torch.long, device=bottleneck_vec.device)
        
        outputs = []
        for t in range(seq_len):
            input_emb = self.embed(input_token).squeeze(1)  # [B, E]
            output, (hidden, cell) = self.decoder(input_emb.unsqueeze(1), (hidden, cell))  # [B,1,H]
            logits = self.output_layer(output.squeeze(1))  # [B, vocab_size]
            outputs.append(logits.unsqueeze(1))
            
            # 采样或Teacher Forcing
            if targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
                input_token = targets[:, t].unsqueeze(1)  # 用真实字符
            else:
                input_token = logits.argmax(dim=1, keepdim=True)  # 用预测字符
        
        outputs = torch.cat(outputs, dim=1)  # [B, L, vocab_size]
        return outputs
    
    def forward(self, x, targets=None, teacher_forcing_ratio=0.5):
        bottleneck_vec = self.encode(x)
        outputs = self.decode(bottleneck_vec, targets, teacher_forcing_ratio)
        return outputs

# --------- 简单训练示例 ---------

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch_x in dataloader:
        batch_x = batch_x.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x, targets=batch_x, teacher_forcing_ratio=0.5)
        loss = criterion(outputs.view(-1, outputs.size(-1)), batch_x.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# --------- 简单数据准备 ---------

# 假设有个小词表，只示意：字符->索引映射
char2idx = {'<pad>':0, '挖':1, '掘':2, '机':3}
idx2char = {v:k for k,v in char2idx.items()}

def encode_text(text, max_len=5):
    # 把文本转为index序列，不足padding
    ids = [char2idx.get(ch,0) for ch in text]
    if len(ids) < max_len:
        ids += [0]*(max_len - len(ids))
    return torch.tensor(ids)

# 训练集只有“挖掘机”这个词的重复示例
train_data = [encode_text("挖掘机")] * 1000

from torch.utils.data import DataLoader
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# --------- 训练 ---------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CharAutoencoder(vocab_size=len(char2idx)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(ignore_index=0)

for epoch in range(10):
    loss = train_epoch(model, train_loader, optimizer, criterion, device)
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

# --------- 测试重构 ---------

model.eval()
test_seq = encode_text("挖掘机").unsqueeze(0).to(device)
with torch.no_grad():
    outputs = model(test_seq, teacher_forcing_ratio=0.0)  # 预测模式，完全靠模型输出

pred_ids = outputs.argmax(dim=-1).squeeze(0).cpu().tolist()
pred_chars = [idx2char.get(i, '') for i in pred_ids]
print("预测重构:", "".join(pred_chars).strip('<pad>'))
