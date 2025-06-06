import torch
from data_process import idx2char,char2idx
from train import device
from model import CharAutoencoder

model = CharAutoencoder(vocab_size=len(char2idx)).to(device)
model.device = device
state_dict = torch.load("./test1/checkpoint/char_autoencoder.pth", map_location=device)
clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
model.load_state_dict(clean_state_dict)
model.eval()

with torch.no_grad():
    outputs = model.load_and_decode("./test1/data/vector.txt", teacher_forcing_ratio=0.5)

pred_ids = outputs.argmax(dim=-1).squeeze(0).cpu().tolist()
pred_chars = [idx2char.get(i, '') for i in pred_ids]
print("预测重构:", "".join(pred_chars).strip('<pad>'))