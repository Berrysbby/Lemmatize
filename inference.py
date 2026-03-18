import torch
import re
from preprocessing import normalize, PUNCT


def predict_sentence(sentence, model, word2idx, char2idx,
                     idx2tag, idx2lemma, device):
    model.eval()

    # Токенизация: убираем пунктуацию
    raw_tokens = sentence.split()
    tokens = []
    for tok in raw_tokens:
        if tok and tok[-1] in PUNCT:
            tokens.append(tok[:-1])
        else:
            if tok not in PUNCT:  # пропускаем отдельные знаки
                tokens.append(tok)

    word_ids = torch.tensor(
        [[word2idx.get(normalize(w), 1) for w in tokens]]
    ).to(device)

    max_word_len = 20
    char_ids = []
    for w in tokens:
        cids = [char2idx.get(c, 1) for c in normalize(w)[:max_word_len]]
        cids += [0] * (max_word_len - len(cids))
        char_ids.append(cids)
    char_ids = torch.tensor([char_ids]).to(device)

    with torch.no_grad():
        tag_logits, lemma_logits = model(word_ids, char_ids)

    tag_preds = tag_logits[0].argmax(-1).cpu().tolist()
    lemma_preds = lemma_logits[0].argmax(-1).cpu().tolist()

    result = []
    for tok, t_id, l_id in zip(tokens, tag_preds, lemma_preds):
        tag = idx2tag.get(t_id, 'UNKN')
        lemma = idx2lemma.get(l_id, normalize(tok))  # fallback: сам токен
        result.append(f"{tok}{{{lemma}={tag}}}")

    return ' '.join(result)