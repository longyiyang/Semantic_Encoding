import torch
from torch.utils.data import DataLoader, Dataset
from collections import Counter

def build_vocab_from_file(file_path, min_freq=1):
    counter = Counter()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            counter.update(line.strip())

    chars = ['<pad>', '<sos>', '<eos>'] + [char for char, freq in counter.items() if freq >= min_freq]
    char2idx = {char: idx for idx, char in enumerate(chars)}
    idx2char = {idx: char for char, idx in char2idx.items()}
    return char2idx, idx2char

def encode_text(text, char2idx, max_len=50):
    ids = [char2idx.get('<sos>')]
    ids += [char2idx.get(ch, char2idx['<pad>']) for ch in text]
    ids.append(char2idx.get('<eos>'))
    if len(ids) < max_len:
        ids += [char2idx.get('<pad>')] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
        ids[-1] = char2idx['<eos>']  #添加关键修改
    return torch.tensor(ids)

class TextDataset(Dataset):
    def __init__(self, file_path, char2idx, max_len=50):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.lines = [line.strip() for line in f if line.strip()]
        self.char2idx = char2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        seq = encode_text(self.lines[idx], self.char2idx, self.max_len)
        return seq, seq  # 输入和目标一样

file_path = './Transformer/data/triple-data.txt'  
max_len = 50    # 每条样本的最大字符长度

# 构建词表
char2idx, idx2char = build_vocab_from_file(file_path)

# 构建 Dataset 和 DataLoader，训练时可调整批大小
dataset = TextDataset(file_path, char2idx, max_len)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)