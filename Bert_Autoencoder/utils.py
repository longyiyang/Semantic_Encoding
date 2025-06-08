import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import tokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import random

MAX_LEN = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEACHER_FORCING_RATIO = 0.7


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
            # 直接用 tgt_ids 作为 decoder 输入，目标就是 tgt_ids
            logits, _ = model(input_ids, attn_mask, tgt_ids)
            target = tgt_ids
        else:
            # 非 teacher forcing，decoder 输入为 input_ids 去掉最后一个 token
            # 目标是 input_ids 向右移动一个 token
            decoder_input = input_ids[:, :-1]
            target = input_ids[:, 1:]
            logits, _ = model(input_ids, attn_mask, decoder_input)

        # 计算 loss，logits 和 target 都对齐后展开
        loss = criterion(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
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