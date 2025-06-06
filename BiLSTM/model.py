import torch.nn as nn
import torch
import numpy as np

class CharAutoencoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, enc_hidden=128, dec_hidden=128, bottleneck_dim=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        
        # 编码器：2层双向LSTM
        self.encoder = nn.LSTM(embed_dim, enc_hidden, num_layers=2, batch_first=True, bidirectional=True)
        
        # 非线性瓶颈降维
        self.bottleneck = nn.Sequential(
            nn.Linear(enc_hidden*2, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Linear(16, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim)
        )
        
        # 解码器输入由瓶颈映射到隐藏状态初始值
        self.bottleneck_to_dec = nn.Linear(bottleneck_dim, dec_hidden)
        
        # 解码器LSTM
        self.decoder = nn.LSTM(embed_dim, dec_hidden,num_layers=2, batch_first=True)
        
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
        
        init_state = self.bottleneck_to_dec(bottleneck_vec)
        hidden = init_state.unsqueeze(0).repeat(self.decoder.num_layers,1,1)
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
                input_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)  # 用预测字符
        
        outputs = torch.cat(outputs, dim=1)  # [B, L, vocab_size]
        return outputs
    
    def save_bottleneck_vec(self, x, path="bottleneck_vec.txt"):
        with torch.no_grad():
            bottleneck_vec = self.encode(x)  # shape: (batch_size, feature_dim)
            np.savetxt(path, bottleneck_vec.cpu().numpy(), fmt="%.6f")
    
    def load_and_decode(self, txt_path, targets=None, teacher_forcing_ratio=0.5):
        vec_np = np.loadtxt(txt_path)  # shape: (batch_size, feature_dim)
        if vec_np.ndim == 1:  # single vector, add batch dimension
            vec_np = vec_np[np.newaxis, :]
        bottleneck_vec = torch.tensor(vec_np, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.decode(bottleneck_vec, targets, teacher_forcing_ratio)
        return outputs
    
    def forward(self, x, targets=None, teacher_forcing_ratio=0.5):
        bottleneck_vec = self.encode(x)
        print("bottleneck_vec:",bottleneck_vec)
        outputs = self.decode(bottleneck_vec, targets, teacher_forcing_ratio)
        return outputs
