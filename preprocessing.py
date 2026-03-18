import re

# Знаки препинания, которые могут встречаться в задаче
PUNCT = set('.,?!')

def tokenize(sentence):
    """Разбивает предложение на токены, отделяя пунктуацию"""
    tokens = sentence.split()
    result = []
    for tok in tokens:
        # Отделяем знаки препинания в конце/начале токена
        if tok and tok[-1] in PUNCT:
            result.append(tok[:-1])
            result.append(tok[-1])
        else:
            result.append(tok)
    return [t for t in result if t]  # убираем пустые

def normalize(word):
    """е/ё нормализация, нижний регистр"""
    return word.lower().replace('ё', 'е')

# Построение словарей
def build_vocab(sentences):
    word2idx = {'<PAD>': 0, '<UNK>': 1}
    char2idx = {'<PAD>': 0, '<UNK>': 1}
    tag2idx = {}
    lemma2idx = {'<PAD>': 0, '<UNK>': 1}

    for sent in sentences:
        for word, lemma, tag in sent:
            w = normalize(word)
            if w not in word2idx:
                word2idx[w] = len(word2idx)
            for ch in w:
                if ch not in char2idx:
                    char2idx[ch] = len(char2idx)
            if lemma not in lemma2idx:
                lemma2idx[lemma] = len(lemma2idx)
            if tag not in tag2idx:
                tag2idx[tag] = len(tag2idx)

    return word2idx, char2idx, tag2idx, lemma2idx