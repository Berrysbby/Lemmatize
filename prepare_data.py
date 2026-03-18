import xml.etree.ElementTree as ET

# Парсим OpenCorpora (корпус в формате XML)
def parse_opencorpora(path):
    tree = ET.parse(path)
    root = tree.getroot()
    sentences = []
    for sent in root.iter('sentence'):
        tokens = []
        for token in sent.iter('token'):
            word = token.attrib['text']
            lem = token.find('.//l')
            if lem is not None:
                lemma = lem.attrib['t'].lower()
                gram = lem.find('g')
                tag = gram.attrib['v'] if gram is not None else 'UNKN'
                # Упрощаем теги до нужных категорий
                tag = simplify_tag(tag)
                tokens.append((word, lemma, tag))
        if tokens:
            sentences.append(tokens)
    return sentences

def simplify_tag(full_tag):
    # OpenCorpora теги -> упрощённые теги задачи
    mapping = {
        'NOUN': 'S', 'ADJF': 'A', 'ADJS': 'A',
        'VERB': 'V', 'INFN': 'V', 'PRTF': 'A', 'PRTS': 'A',
        'ADVB': 'ADV', 'PRED': 'ADV',
        'NPRO': 'NI', 'NUMR': 'NUM', 'PREP': 'PR',
        'CONJ': 'CONJ', 'PRCL': 'PART', 'INTJ': 'INTJ',
    }
    for k, v in mapping.items():
        if full_tag.startswith(k):
            return v
    return 'UNKN'