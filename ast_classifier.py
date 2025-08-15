import torch
import torch.nn as nn

LABELS = {
    "none": 0,
    "syntactic": 1,
    "conceptual": 2,
    "strategic": 3
}

class ASTClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_classes=4):
        super(ASTClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # 双向GRU输出维度翻倍

    def forward(self, x):
        embed = self.embedding(x)  # (batch, seq_len, embed_dim)
        output, _ = self.encoder(embed)  # output: (batch, seq_len, hidden*2)
        pooled = torch.mean(output, dim=1)  # 对序列取平均池化
        pooled = self.dropout(pooled)
        out = self.fc(pooled)  # (batch, num_classes)
        return out


def tokenize(code_snippet, vocab):

    import re
    tokens = re.findall(r"\w+|[^\s\w]", code_snippet)
    return [vocab.get(tok, vocab.get("<unk>", 0)) for tok in tokens]











# import torch
# import torch.nn as nn
#
# LABELS = {
#     "none": 0,
#     "syntactic": 1,
#     "conceptual": 2,
#     "strategic": 3
# }
#
# class ASTClassifier(nn.Module):
#     def __init__(self, vocab_size, embed_dim=64, hidden_dim=64, num_classes=4):
#         super(ASTClassifier, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embed_dim)
#         self.encoder = nn.GRU(embed_dim, hidden_dim, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, num_classes)
#
#     def forward(self, x):
#         embed = self.embedding(x)
#         _, h = self.encoder(embed)
#         out = self.fc(h.squeeze(0))
#         return out
#
#
# def tokenize(code_snippet, vocab):
#     """将代码按空格和符号拆分为 token 序列，并映射为 vocab index"""
#     import re
#     tokens = re.findall(r"\w+|[^\s\w]", code_snippet)
#     return [vocab.get(tok, vocab.get("<unk>", 0)) for tok in tokens]
