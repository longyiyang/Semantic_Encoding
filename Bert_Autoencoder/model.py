import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torch.nn import TransformerDecoder, TransformerDecoderLayer

LATENT_DIM = 6
MAX_LEN = 50
EMBED_DIM = 768
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained("./Bert_Autoencoder/bert/bert-base-chinese")

class BertAutoEncoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, max_len=MAX_LEN):
        super().__init__()
        self.bert = BertModel.from_pretrained("./Bert_Autoencoder/bert/bert-base-chinese")
        self.to_latent = nn.Linear(EMBED_DIM, latent_dim)
        self.latent_to_memory = nn.Linear(latent_dim, EMBED_DIM)

        decoder_layer = TransformerDecoderLayer(d_model=EMBED_DIM, nhead=8)
        self.decoder = TransformerDecoder(decoder_layer, num_layers=3)

        self.output_proj = nn.Linear(EMBED_DIM, self.bert.config.vocab_size)
        self.positional_encoding = nn.Parameter(torch.randn(max_len, EMBED_DIM))

    def forward(self, input_ids, attention_mask, tgt_ids):
        with torch.no_grad():
            encoder_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls = encoder_outputs.last_hidden_state[:, 0]

        latent = self.to_latent(cls)
        memory = self.latent_to_memory(latent).unsqueeze(0)

        tgt_embeddings = self.bert.embeddings(tgt_ids) + self.positional_encoding[:tgt_ids.size(1)].unsqueeze(0)
        decoded = self.decoder(tgt=tgt_embeddings.transpose(0, 1), memory=memory)

        logits = self.output_proj(decoded.transpose(0, 1))
        return logits, latent

    def encode(self, input_ids, attention_mask):
        with torch.no_grad():
            encoder_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls = encoder_outputs.last_hidden_state[:, 0]
        return self.to_latent(cls)

    def decode(self, latent, max_len=100, beam_width=3):
        self.eval()
        with torch.no_grad():
            memory = self.latent_to_memory(latent).unsqueeze(0)
            sequences = [[[], 0.0]]  # (sequence, score)

            for step in range(max_len):
                all_candidates = []
                for seq, score in sequences:
                    # 如果已经生成了eos_token，直接保留，不扩展
                    if len(seq) > 0 and seq[-1] == tokenizer.eos_token_id:
                        all_candidates.append([seq, score])
                        continue

                    ids = [tokenizer.cls_token_id] + seq
                    tgt = torch.tensor([ids], device=DEVICE)
                    if tgt.shape[1] >= max_len:
                        all_candidates.append([seq, score])
                        continue

                    tgt_embed = self.bert.embeddings(tgt) + self.positional_encoding[:tgt.size(1)].unsqueeze(0)
                    out = self.decoder(tgt=tgt_embed.transpose(0, 1), memory=memory)
                    logits = self.output_proj(out.transpose(0, 1))[:, -1, :]
                    probs = torch.softmax(logits, dim=-1)

                    topk = torch.topk(probs, min(beam_width, probs.size(-1)))

                    # 如果topk的概率都极低，可以考虑停止
                    if topk.values.max().item() < 1e-5:
                        print(f"Low max prob at step {step}, stopping expansion for this seq")
                        continue

                    for i in range(topk.indices.size(1)):
                        token_id = topk.indices[0][i].item()
                        prob = topk.values[0][i].item()
                        new_score = score - torch.log(torch.tensor(prob + 1e-12)).item()
                        all_candidates.append([seq + [token_id], new_score])

                if len(all_candidates) == 0:
                    print(f"No candidates generated at step {step}, stopping beam search.")
                    break

                sequences = sorted(all_candidates, key=lambda x: x[1])[:beam_width]

            if len(sequences) == 0:
                print("No sequences generated, returning empty string")
                return ""

            best_seq = sequences[0][0]
            return tokenizer.decode(best_seq, skip_special_tokens=True)

