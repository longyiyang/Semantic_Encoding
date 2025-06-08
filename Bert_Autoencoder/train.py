from model import BertAutoEncoder,tokenizer
from utils import SentenceDataset,DataLoader,LabelSmoothingLoss,evaluate_bleu,train
import torch 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 5
SMOOTHING = 0.1
MODEL_PATH = "./Bert_Autoencoder/checkpoint/bert_autoencoder.pt"

# ========== 示例语料 ==========
with open("./Bert_Autoencoder/data/data.txt", "r", encoding="utf-8") as f:
    sample_sentences = [line.strip() for line in f if line.strip()]

# ========== 初始化 ==========
model = BertAutoEncoder().to(DEVICE)
dataset = SentenceDataset(sample_sentences)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = LabelSmoothingLoss(classes=tokenizer.vocab_size, smoothing=SMOOTHING, ignore_index=tokenizer.pad_token_id)

# ========== 训练 ==========
for epoch in range(EPOCHS):
    loss = train(model, dataloader, optimizer, criterion)
    #bleu = evaluate_bleu(model, dataset)
    # "| BLEU: {bleu:.4f}"
    print(f"Epoch {epoch+1} | Loss: {loss:.4f} ")

# ========== 保存模型 =========
torch.save(model.state_dict(), MODEL_PATH)