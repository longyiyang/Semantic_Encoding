import torch 
import torch.nn as nn
from model import CharAutoencoder
from data_process import char2idx,train_loader

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch_x in dataloader:
        batch_x = batch_x.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x, targets=batch_x, teacher_forcing_ratio=0.5)
        loss = criterion(outputs.view(-1, outputs.size(-1)), batch_x.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CharAutoencoder(vocab_size=len(char2idx)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(ignore_index=0)

if __name__ == "__main__":
    for epoch in range(50):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
    torch.save(model.state_dict(), "./test1/checkpoint/char_autoencoder.pth")