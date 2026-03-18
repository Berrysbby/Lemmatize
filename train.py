import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from preprocessing import normalize, PUNCT

class MorphDataset(Dataset):
    def __init__(self, sentences, word2idx, char2idx, tag2idx, lemma2idx,
                 max_word_len=20):
        self.data = []
        self.max_word_len = max_word_len

        for sent in sentences:
            words, lemmas, tags = zip(*sent)

            word_ids = [word2idx.get(normalize(w), 1) for w in words]

            char_ids = []
            for w in words:
                cids = [char2idx.get(c, 1) for c in normalize(w)[:max_word_len]]
                # padding до max_word_len
                cids += [0] * (max_word_len - len(cids))
                char_ids.append(cids)

            tag_ids = [tag2idx.get(t, 0) for t in tags]
            lemma_ids = [lemma2idx.get(l, 1) for l in lemmas]

            self.data.append((word_ids, char_ids, tag_ids, lemma_ids))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    """Паддинг батча до одинаковой длины"""
    max_len = max(len(item[0]) for item in batch)
    max_wlen = len(batch[0][1][0])

    word_ids, char_ids, tag_ids, lemma_ids, masks = [], [], [], [], []
    for words, chars, tags, lemmas in batch:
        pad_len = max_len - len(words)
        word_ids.append(words + [0] * pad_len)
        char_ids.append(chars + [[0] * max_wlen] * pad_len)
        tag_ids.append(tags + [0] * pad_len)
        lemma_ids.append(lemmas + [0] * pad_len)
        masks.append([1] * len(words) + [0] * pad_len)

    return (torch.tensor(word_ids), torch.tensor(char_ids),
            torch.tensor(tag_ids), torch.tensor(lemma_ids),
            torch.tensor(masks, dtype=torch.bool))


def train_epoch(model, loader, optimizer, device):
    model.train()
    tag_criterion = nn.CrossEntropyLoss(ignore_index=0)
    lemma_criterion = nn.CrossEntropyLoss(ignore_index=0)
    total_loss = 0

    for word_ids, char_ids, tag_ids, lemma_ids, mask in loader:
        word_ids, char_ids = word_ids.to(device), char_ids.to(device)
        tag_ids, lemma_ids = tag_ids.to(device), lemma_ids.to(device)

        optimizer.zero_grad()
        tag_logits, lemma_logits = model(word_ids, char_ids)

        # (B*S, vocab_size) для CrossEntropy
        B, S, _ = tag_logits.shape
        tag_loss = tag_criterion(tag_logits.view(B * S, -1), tag_ids.view(B * S))
        lemma_loss = lemma_criterion(lemma_logits.view(B * S, -1), lemma_ids.view(B * S))

        loss = tag_loss + lemma_loss  # multi-task loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)