import torch 
import numpy as np
from model import CharAutoencoder
from data_process import encode_text,char2idx 
from train import device

model = CharAutoencoder(vocab_size=len(char2idx)).to(device)
model.device=device
state_dict = torch.load("./test1/checkpoint/char_autoencoder.pth", map_location=device)
clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
model.load_state_dict(clean_state_dict)
model.eval()

x = encode_text("道路拥挤 传达 J9区域").unsqueeze(0).to(device)
model.save_bottleneck_vec(x,"./test1/data/bottleneck_vec.txt")

