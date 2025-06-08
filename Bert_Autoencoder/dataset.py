from transformers import BertTokenizer
from torch.utils.data import Dataset
import torch

tokenizer = BertTokenizer.from_pretrained('./Bert_Autoencoder/bert/bert-base-chinese')
MAX_LEN = 50

class ChineseTextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]

        encoded = tokenizer.encode_plus(
            text,
            padding='max_length',
            truncation=True,
            max_length=MAX_LEN,
            return_tensors='pt'
        )

        tokens = tokenizer.tokenize(text)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        token_ids = token_ids[:MAX_LEN-2]
        token_ids = [tokenizer.cls_token_id] + token_ids + [tokenizer.sep_token_id]
        pad_len = MAX_LEN - len(token_ids)
        decoder_input_ids = token_ids[:-1] + [tokenizer.pad_token_id] * pad_len
        decoder_target_ids = token_ids[1:] + [tokenizer.pad_token_id] * pad_len

        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'decoder_input_ids': torch.tensor(decoder_input_ids),
            'decoder_target_ids': torch.tensor(decoder_target_ids),
        }
