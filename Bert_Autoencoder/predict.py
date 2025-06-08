import torch
from transformers import BertTokenizer
from model import TextAutoencoder

tokenizer = BertTokenizer.from_pretrained('./Bert_Autoencoder/bert/bert-base-chinese')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TextAutoencoder()
model.load_state_dict(torch.load("./Bert_Autoencoder/checkpoint/text_autoencoder.pt", map_location=device))
model.to(device)
model.eval()

def generate_text(text, max_len=50):
    with torch.no_grad():
        tokens = tokenizer.encode(text, return_tensors='pt').to(device)
        mask = tokens != tokenizer.pad_token_id

        encoder_output = model.encoder(tokens, attention_mask=mask)
        cls_embedding = encoder_output.last_hidden_state[:, 0, :]
        latent = model.projection(cls_embedding)
        memory = model.decoder_input_proj(latent).unsqueeze(1).transpose(0, 1)

        next_input = torch.tensor([[tokenizer.cls_token_id]], device=device)
        output_tokens = []

        for _ in range(max_len):
            tgt_embed = model.token_embedding(next_input).transpose(0, 1)
            out = model.transformer_decoder(tgt_embed, memory)
            logits = model.output_proj(out.transpose(0, 1))[:, -1, :]
            next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
            output_tokens.append(next_token.item())
            if next_token.item() == tokenizer.sep_token_id:
                break
            next_input = torch.cat([next_input, next_token], dim=1)

        return tokenizer.decode(output_tokens)

# 示例
print(generate_text("今天的天气怎么样"))
