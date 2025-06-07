import torch
from torch.utils.data import DataLoader, Dataset

def build_char2idx_from_file(filepath, add_pad=True):
    char_set = set()
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # 非空行
                char_set.update(line)  # 添加所有字符
    
    char_list = sorted(char_set)  # 可排序以保证一致性，也可不排序
    char2idx = {'<pad>': 0} if add_pad else {}
    
    start_idx = 1 if add_pad else 0
    for i, char in enumerate(char_list, start=start_idx):
        char2idx[char] = i

    return char2idx


# 假设有个小词表，只示意：字符->索引映射
char2idx = build_char2idx_from_file('./BiLSTM/data/entities.txt')
idx2char = {v:k for k,v in char2idx.items()}

def encode_text(text, max_len=50):
    # 把文本转为index序列，不足padding
    ids = [char2idx.get(ch,0) for ch in text]
    if len(ids) < max_len:
        ids += [0]*(max_len - len(ids))
    return torch.tensor(ids)

class EntityDataset(Dataset):
    def __init__(self, filepath, max_len=50):
        self.samples = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip()
                if text:
                    self.samples.append(encode_text(text, max_len))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

#创建 DataLoader
train_dataset = EntityDataset('./BiLSTM/data/entities.txt', max_len=50)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)