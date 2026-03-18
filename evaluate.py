from inference import predict_sentence
from preprocessing import normalize

def evaluate(model, test_sentences, word2idx, char2idx,
             idx2tag, idx2lemma, device):
    tag_correct, lemma_correct, total = 0, 0, 0

    for sent in test_sentences:
        words, true_lemmas, true_tags = zip(*sent)
        pred_output = predict_sentence(
            ' '.join(words), model, word2idx, char2idx,
            idx2tag, idx2lemma, device
        )
        # парсим предсказания
        pred_tokens = pred_output.split()
        for i, pt in enumerate(pred_tokens):
            if i >= len(true_tags):
                break
            # извлекаем лемму и тег из "токен{лемма=тег}"
            import re
            m = re.search(r'\{(.+)=(.+)\}', pt)
            if m:
                pred_lemma, pred_tag = m.group(1), m.group(2)
                if pred_tag == true_tags[i]:
                    tag_correct += 1
                if normalize(pred_lemma) == normalize(true_lemmas[i]):
                    lemma_correct += 1
                total += 1

    print(f"POS Accuracy:   {tag_correct/total:.4f}")
    print(f"Lemma Accuracy: {lemma_correct/total:.4f}")