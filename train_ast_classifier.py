import torch
import torch.nn as nn
import torch.optim as optim
import json
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from ast_classifier import ASTClassifier, tokenize, LABELS
import random

# ==== Configuration ====
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-3
MAX_LEN = 50

data_path = "dataset.json"
vocab_path = "vocab.json"
model_path = "ast_model.pt"

# ==== Load data ====
with open(data_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

all_codes = [item["code"] for item in raw_data]
all_labels = [LABELS[item["label"]] for item in raw_data]

# ==== Build vocabulary ====
all_tokens = set()
for code in all_codes:
    tokens = tokenize(code, {})
    all_tokens.update(tokens)

token_to_id = {tok: idx + 1 for idx, tok in enumerate(sorted(all_tokens))}
token_to_id["<unk>"] = 0

# Save vocab.json
with open(vocab_path, "w", encoding="utf-8") as f:
    json.dump(token_to_id, f, indent=2, ensure_ascii=False)

# ==== Dataset wrapper ====
class ASTDataset(Dataset):
    def __init__(self, codes, labels, vocab):
        self.codes = codes
        self.labels = labels
        self.vocab = vocab

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        code = self.codes[idx]
        tokens = tokenize(code, self.vocab)
        tokens = tokens[:MAX_LEN] + [0] * (MAX_LEN - len(tokens))
        return torch.tensor(tokens), torch.tensor(self.labels[idx])

# ==== Split into training and validation sets ====
train_codes, val_codes, train_labels, val_labels = train_test_split(all_codes, all_labels, test_size=0.2, random_state=42)

train_set = ASTDataset(train_codes, train_labels, token_to_id)
val_set = ASTDataset(val_codes, val_labels, token_to_id)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

# ==== Build model ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ASTClassifier(vocab_size=len(token_to_id), embed_dim=64, hidden_dim=64, num_classes=len(LABELS)).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# ==== Train ====
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    acc = correct / total
    print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f} | Val Acc: {acc:.2%}")

# ==== Save model ====
torch.save(model.state_dict(), model_path)
print(f" Model saved to {model_path}")
