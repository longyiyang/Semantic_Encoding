import torch 
import torch.nn as nn
from data_process import train_loader
from model import TransformerAutoencoder
from data_process import char2idx

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerAutoencoder(vocab_size=len(char2idx), max_len=50).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(ignore_index=0)

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch_input, batch_target in dataloader:
        batch_input = batch_input.to(device)
        batch_target = batch_target.to(device)

        tgt_input = batch_input[:, :-1]   # 解码器输入，去掉 <eos>
        tgt_output = batch_target[:, 1:]  # 目标，去掉 <sos>

        optimizer.zero_grad()
        outputs = model(batch_input, tgt_input)
        loss = criterion(outputs.reshape(-1, outputs.size(-1)), tgt_output.reshape(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

if __name__ == "__main__":
    for epoch in range(10):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
    torch.save(model.state_dict(), "./Transformer/checkpoint/char_autoencoder.pth")