import torch
from data_process import encode_text,idx2char,char2idx
from train import model,device
from model import CharAutoencoder

model = CharAutoencoder(vocab_size=len(char2idx)).to(device)
model.load_state_dict(torch.load("./test1/checkpoint/char_autoencoder.pth", map_location=device))
model.eval()
test_seq = encode_text("范小勤 报告 雷击").unsqueeze(0).to(device)
with torch.no_grad():
    outputs = model(test_seq, teacher_forcing_ratio=0.0)  # 预测模式，完全靠模型输出

pred_ids = outputs.argmax(dim=-1).squeeze(0).cpu().tolist()
pred_chars = [idx2char.get(i, '') for i in pred_ids]
print("预测重构:", "".join(pred_chars).strip('<pad>'))