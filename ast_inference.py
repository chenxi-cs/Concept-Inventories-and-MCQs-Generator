# ast_inference.py

import torch
from ast_classifier import ASTClassifier, tokenize, LABELS
import json

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load vocab
with open("vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

# Load model
model = ASTClassifier(
    vocab_size=len(vocab), embed_dim=64, hidden_dim=64, num_classes=4
)
model.load_state_dict(torch.load("ast_model.pt", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Inverse label lookup
INV_LABELS = {v: k for k, v in LABELS.items()}


def detect_misconception_nodes(code_snippet: str):
    """
    Input a Java code snippet, output potential structural misconceptions (simplified implementation)
    """
    results = []

    lines = code_snippet.split(";")  # Roughly split substructures
    for fragment in lines:
        fragment = fragment.strip()
        if not fragment:
            continue
        tokens = tokenize(fragment, vocab, 32)
        input_tensor = torch.tensor([tokens]).to(DEVICE)
        with torch.no_grad():
            logits = model(input_tensor)
            pred = torch.argmax(logits, dim=1).item()
            label = INV_LABELS[pred]
            if label != "none":
                results.append({"subtree": fragment, "type": label})
    return results
