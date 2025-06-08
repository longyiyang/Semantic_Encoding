import torch
from model import tokenizer,BertAutoEncoder

MAX_LEN = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "./Bert_Autoencoder/checkpoint/bert_autoencoder.pt"

# ========== 测试重建 ==========
model = BertAutoEncoder().to(DEVICE)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
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
