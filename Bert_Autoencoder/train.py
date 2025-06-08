import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from transformers import BertTokenizer
from model import TextAutoencoder
from dataset import ChineseTextDataset

def load_text_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


tokenizer = BertTokenizer.from_pretrained('./Bert_Autoencoder/bert/bert-base-chinese')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

texts = load_text_file("./Bert_Autoencoder/data/data.txt")
dataset = ChineseTextDataset(texts)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

model = TextAutoencoder().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

EPOCHS = 5
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        decoder_input_ids = batch['decoder_input_ids'].to(device)
        decoder_target_ids = batch['decoder_target_ids'].to(device)

        logits = model(input_ids, attention_mask, decoder_input_ids)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            decoder_target_ids.view(-1),
            ignore_index=tokenizer.pad_token_id
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"[Epoch {epoch+1}] Loss: {total_loss / len(dataloader):.4f}")

torch.save(model.state_dict(), "./Bert_Autoencoder/checkpoint/text_autoencoder.pt")
