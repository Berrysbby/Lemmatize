import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os

from prepare_data import parse_opencorpora
from preprocessing import build_vocab
from model import LemmaTaggerModel
from train import MorphDataset, collate_fn, train_epoch
from inference import predict_sentence
from evaluate import evaluate

# ──────────────────────────────────────────
# 1. Конфигурация
# ──────────────────────────────────────────
CORPUS_PATH  = "C:/Users/veron/Documents/unik/lemmatize/annot.opcorpora.xml"   # путь к файлу OpenCorpora
EPOCHS       = 10
BATCH_SIZE   = 32
LR           = 1e-3
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH   = "morph_model.pt"

# ──────────────────────────────────────────
# 2. Загрузка и разбивка данных
# ──────────────────────────────────────────
print("Загрузка корпуса...")
sentences = parse_opencorpora(CORPUS_PATH)

train_size = int(0.9 * len(sentences))
train_sents = sentences[:train_size]
test_sents  = sentences[train_size:]
print(f"Train: {len(train_sents)}, Test: {len(test_sents)}")

# ──────────────────────────────────────────
# 3. Построение словарей
# ──────────────────────────────────────────
word2idx, char2idx, tag2idx, lemma2idx = build_vocab(train_sents)

idx2tag   = {v: k for k, v in tag2idx.items()}
idx2lemma = {v: k for k, v in lemma2idx.items()}

print(f"Vocab: words={len(word2idx)}, chars={len(char2idx)}, "
      f"tags={len(tag2idx)}, lemmas={len(lemma2idx)}")

# ──────────────────────────────────────────
# 4. DataLoader
# ──────────────────────────────────────────
train_dataset = MorphDataset(train_sents, word2idx, char2idx, tag2idx, lemma2idx)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                           shuffle=True, collate_fn=collate_fn)

# ──────────────────────────────────────────
# 5. Модель и оптимизатор
# ──────────────────────────────────────────
model = LemmaTaggerModel(
    word_vocab_size  = len(word2idx),
    char_vocab_size  = len(char2idx),
    tag_vocab_size   = len(tag2idx),
    lemma_vocab_size = len(lemma2idx),
).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=LR)

"""# ──────────────────────────────────────────
# 6. Обучение
# ──────────────────────────────────────────
for epoch in range(1, EPOCHS + 1):
    loss = train_epoch(model, train_loader, optimizer, DEVICE)
    print(f"Epoch {epoch}/{EPOCHS}  loss={loss:.4f}")

# ──────────────────────────────────────────
# 7. Сохранение модели
# ──────────────────────────────────────────
torch.save({
    "model_state": model.state_dict(),
    "word2idx": word2idx,
    "char2idx": char2idx,
    "tag2idx":  tag2idx,
    "lemma2idx": lemma2idx,
}, MODEL_PATH)
print(f"Модель сохранена в {MODEL_PATH}")"""

if os.path.exists(MODEL_PATH):
    print("Загрузка сохранённой модели...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    model.load_state_dict(checkpoint["model_state"])
    word2idx = checkpoint["word2idx"]
    char2idx = checkpoint["char2idx"]
    tag2idx = checkpoint["tag2idx"]
    lemma2idx = checkpoint["lemma2idx"]

    idx2tag   = {v: k for k, v in tag2idx.items()}
    idx2lemma = {v: k for k, v in lemma2idx.items()}

else:
    print("Файл модели не найден.")



# ──────────────────────────────────────────
# 8. Оценка
# ──────────────────────────────────────────
evaluate(model, test_sents, word2idx, char2idx, idx2tag, idx2lemma, DEVICE)

# ──────────────────────────────────────────
# 9. Пример инференса
# ──────────────────────────────────────────
test_input = "Стала стабильнее экономическая и политическая обстановка."
print("\nПример:")
print("Вход:", test_input)
print("Выход:", predict_sentence(
    test_input, model, word2idx, char2idx, idx2tag, idx2lemma, DEVICE
))

# ──────────────────────────────────────────
# 10. Интерактивный режим
# ──────────────────────────────────────────
print("\nИнтерактивный режим. Введите предложение (или 'exit' для выхода).")

while True:
    user_input = input("\n>>> ")
    
    if user_input.lower() == "exit":
        print("Выход из интерактивного режима.")
        break

    if not user_input.strip():
        continue

    output = predict_sentence(
        user_input,
        model,
        word2idx,
        char2idx,
        idx2tag,
        idx2lemma,
        DEVICE
    )

    print(output)