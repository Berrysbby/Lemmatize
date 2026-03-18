import torch
import torch.nn as nn

class CharCNN(nn.Module):
    """Символьные эмбеддинги через CNN"""
    def __init__(self, char_vocab_size, char_emb_dim=30, num_filters=50, kernel_size=3):
        super().__init__()
        self.embedding = nn.Embedding(char_vocab_size, char_emb_dim, padding_idx=0)
        self.conv = nn.Conv1d(char_emb_dim, num_filters, kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, seq_len, max_word_len)
        B, S, W = x.shape
        x = x.view(B * S, W)
        emb = self.embedding(x)             # (B*S, W, char_emb_dim)
        emb = emb.transpose(1, 2)           # (B*S, char_emb_dim, W)
        out = self.relu(self.conv(emb))     # (B*S, num_filters, W)
        out = out.max(dim=2).values         # (B*S, num_filters) — max pooling
        return out.view(B, S, -1)           # (B, S, num_filters)


class LemmaTaggerModel(nn.Module):
    def __init__(self, word_vocab_size, char_vocab_size,
                 tag_vocab_size, lemma_vocab_size,
                 word_emb_dim=100, char_emb_dim=30,
                 char_filters=50, hidden_dim=256, dropout=0.5):
        super().__init__()

        self.word_emb = nn.Embedding(word_vocab_size, word_emb_dim, padding_idx=0)
        self.char_cnn = CharCNN(char_vocab_size, char_emb_dim, char_filters)

        input_dim = word_emb_dim + char_filters

        self.bilstm = nn.LSTM(
            input_dim, hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)

        # Два выходных слоя: для тега и для леммы
        self.tag_head = nn.Linear(hidden_dim, tag_vocab_size)
        self.lemma_head = nn.Linear(hidden_dim, lemma_vocab_size)

    def forward(self, word_ids, char_ids):
        # word_ids: (B, S)
        # char_ids: (B, S, W)
        word_emb = self.word_emb(word_ids)        # (B, S, word_emb_dim)
        char_emb = self.char_cnn(char_ids)        # (B, S, char_filters)

        x = torch.cat([word_emb, char_emb], dim=-1)  # (B, S, input_dim)
        x = self.dropout(x)

        out, _ = self.bilstm(x)                   # (B, S, hidden_dim)
        out = self.dropout(out)

        tag_logits = self.tag_head(out)            # (B, S, tag_vocab_size)
        lemma_logits = self.lemma_head(out)        # (B, S, lemma_vocab_size)

        return tag_logits, lemma_logits