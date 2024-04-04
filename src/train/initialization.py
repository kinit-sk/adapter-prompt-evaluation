from transformers import AutoTokenizer


def init_tokens(tokenizer, size=5000):
    vocab = tokenizer.get_vocab()
    vocab = {k: v for k, v in sorted(vocab.items(), key=lambda item: item[1])}
    for k in list(vocab.keys()):
        if k in tokenizer.all_special_tokens:
            del vocab[k]

    # get token indexes
    token_indexes = list(vocab.values())
    token_indexes = token_indexes[:size]
    return token_indexes


def class_initialization(tokenizer, classes):
    token_indexes = []
    for class_ in classes:
        token_indexes.append(*tokenizer.encode(class_)['input_ids'])

    return token_indexes
