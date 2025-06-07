# ✅ 优化版 PyTorch 自编码器：BERT 编码器 + 6维瓶颈 + Transformer 解码器 + Label Smoothing + Teacher Forcing + Mixed Beam-Greedy Decode

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from torch.utils.data import Dataset, DataLoader
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import random
import os

# ========== 超参数 ==========
LATENT_DIM = 6
MAX_LEN = 20
EMBED_DIM = 768
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
EPOCHS = 10
MODEL_PATH = "bert_autoencoder.pt"
SMOOTHING = 0.1
TEACHER_FORCING_RATIO = 0.7

# ========== 中文 Tokenizer（HuggingFace） ==========
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

# ========== 构造模型 ==========
class BertAutoEncoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, max_len=MAX_LEN):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-chinese")
        self.to_latent = nn.Linear(EMBED_DIM, latent_dim)
        self.latent_to_memory = nn.Linear(latent_dim, EMBED_DIM)

        decoder_layer = TransformerDecoderLayer(d_model=EMBED_DIM, nhead=8)
        self.decoder = TransformerDecoder(decoder_layer, num_layers=3)

        self.output_proj = nn.Linear(EMBED_DIM, self.bert.config.vocab_size)
        self.positional_encoding = nn.Parameter(torch.randn(max_len, EMBED_DIM))

    def forward(self, input_ids, attention_mask, tgt_ids):
        with torch.no_grad():
            encoder_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls = encoder_outputs.last_hidden_state[:, 0]

        latent = self.to_latent(cls)
        memory = self.latent_to_memory(latent).unsqueeze(0)

        tgt_embeddings = self.bert.embeddings(tgt_ids) + self.positional_encoding[:tgt_ids.size(1)].unsqueeze(0)
        decoded = self.decoder(tgt=tgt_embeddings.transpose(0, 1), memory=memory)

        logits = self.output_proj(decoded.transpose(0, 1))
        return logits, latent

    def encode(self, input_ids, attention_mask):
        with torch.no_grad():
            encoder_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls = encoder_outputs.last_hidden_state[:, 0]
        return self.to_latent(cls)

    def decode(self, latent, max_len=MAX_LEN, beam_width=3):
        self.eval()
        with torch.no_grad():
            memory = self.latent_to_memory(latent).unsqueeze(0)
            sequences = [[list(), 0.0]]

            for step in range(max_len):
                all_candidates = []
                for seq, score in sequences:
                    ids = [tokenizer.cls_token_id] + seq
                    tgt = torch.tensor([ids], device=DEVICE)
                    if tgt.shape[1] >= max_len:
                        continue
                    tgt_embed = self.bert.embeddings(tgt) + self.positional_encoding[:tgt.size(1)].unsqueeze(0)
                    out = self.decoder(tgt=tgt_embed.transpose(0, 1), memory=memory)
                    logits = self.output_proj(out.transpose(0, 1))[:, -1, :]
                    probs = torch.softmax(logits, dim=-1)
                    topk = torch.topk(probs, beam_width)
                    for i in range(beam_width):
                        token_id = topk.indices[0][i].item()
                        prob = topk.values[0][i].item()
                        candidate = [seq + [token_id], score - torch.log(torch.tensor(prob + 1e-12)).item()]
                        all_candidates.append(candidate)
                sequences = sorted(all_candidates, key=lambda x: x[1])[:beam_width]

            best_seq = sequences[0][0]
            return tokenizer.decode(best_seq, skip_special_tokens=True)

# ========== 数据集定义 ==========
class SentenceDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        text = self.sentences[idx]
        encoding = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=MAX_LEN)
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        return input_ids, attention_mask, input_ids

# ========== 编码 & 解码工具 ==========
def beam_search_decode(model, input_ids, attention_mask, beam_width=3):
    model.eval()
    with torch.no_grad():
        latent = model.encode(input_ids, attention_mask)
        return model.decode(latent, max_len=MAX_LEN, beam_width=beam_width)

# ========== 自定义 Label Smoothing 损失 ==========
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, ignore_index=-100):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.ignore_index = ignore_index

    def forward(self, x, target):
        x = x.log_softmax(dim=-1)
        true_dist = torch.zeros_like(x)
        true_dist.fill_(self.smoothing / (self.cls - 1))
        ignore = target == self.ignore_index
        target = target.masked_fill(ignore, 0)
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist.masked_fill_(ignore.unsqueeze(1), 0)
        return torch.mean(torch.sum(-true_dist * x, dim=-1))

# ========== 模型训练函数 ==========
def train(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for input_ids, attn_mask, tgt_ids in dataloader:
        input_ids, attn_mask, tgt_ids = input_ids.to(DEVICE), attn_mask.to(DEVICE), tgt_ids.to(DEVICE)
        optimizer.zero_grad()

        if random.random() < TEACHER_FORCING_RATIO:
            logits, _ = model(input_ids, attn_mask, tgt_ids)
        else:
            logits, _ = model(input_ids, attn_mask, input_ids[:, :-1])

        loss = criterion(logits.view(-1, logits.size(-1)), tgt_ids.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# ========== 评估函数：BLEU ==========
def evaluate_bleu(model, dataset):
    model.eval()
    total_bleu = 0
    smooth = SmoothingFunction().method1
    with torch.no_grad():
        for input_ids, attn_mask, _ in DataLoader(dataset, batch_size=1):
            input_ids, attn_mask = input_ids.to(DEVICE), attn_mask.to(DEVICE)
            decoded = beam_search_decode(model, input_ids, attn_mask)
            reference = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            total_bleu += sentence_bleu([list(reference)], list(decoded), smoothing_function=smooth)
    return total_bleu / len(dataset)

# ========== 示例语料 ==========
sample_sentences = [
    "我爱自然语言处理",
    "天气真好我们出去玩吧",
    "深度学习正在改变世界",
    "今天天气不错适合散步",
    "机器学习和人工智能有很大关系",
    "生活就像一盒巧克力",
    "你好世界",
    "学习使我快乐"
]

# ========== 初始化 ==========
model = BertAutoEncoder().to(DEVICE)
dataset = SentenceDataset(sample_sentences)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = LabelSmoothingLoss(classes=tokenizer.vocab_size, smoothing=SMOOTHING, ignore_index=tokenizer.pad_token_id)

# ========== 训练 ==========
for epoch in range(EPOCHS):
    loss = train(model, dataloader, optimizer, criterion)
    bleu = evaluate_bleu(model, dataset)
    print(f"Epoch {epoch+1} | Loss: {loss:.4f} | BLEU: {bleu:.4f}")

# ========== 保存模型 ==========
torch.save(model.state_dict(), MODEL_PATH)

# ========== 测试重建 ==========
model.eval()
test_text = "我爱自然语言处理"
encoding = tokenizer(test_text, return_tensors='pt', padding='max_length', truncation=True, max_length=MAX_LEN)
input_ids = encoding['input_ids'].to(DEVICE)
attn_mask = encoding['attention_mask'].to(DEVICE)

with torch.no_grad():
    latent = model.encode(input_ids, attn_mask)
    decoded_text = model.decode(latent)
    print("原句:", test_text)
    print("编码向量 (6维):", latent.squeeze().cpu().numpy())
    print("重建:", decoded_text)
