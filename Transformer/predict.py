import torch
from model import TransformerAutoencoder
from data_process import idx2char, encode_text, char2idx
from train import device

model = TransformerAutoencoder(vocab_size=len(char2idx), max_len=50).to(device)
model.load_state_dict(torch.load("./Transformer/checkpoint/char_autoencoder.pth", map_location=device))
model.eval()

input_text = "通讯小车 报告 VVVV区爆炸物成分"
input_tensor = encode_text(input_text, char2idx).unsqueeze(0).to(device)

sos_idx = char2idx['<sos>']
eos_idx = char2idx['<eos>']

with torch.no_grad():
    output_tensor = model.generate(input_tensor, sos_idx=sos_idx, eos_idx=eos_idx)
    output_ids = output_tensor[0].tolist()

# 截断到 <eos>
if eos_idx in output_ids:
    output_ids = output_ids[:output_ids.index(eos_idx)]

output_text = ''.join([idx2char.get(i, '') for i in output_ids if i not in (sos_idx,)])
print("输入文本：", input_text)
print("模型重建：", output_text)
