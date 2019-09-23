import random

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

from modelconfig import args


def load_vocab(name):
    tokens = []
    with open(name, 'r') as f:
        for token in f:
            tokens.append(token[:-1])
    tokens = {k: (v + 1) for v, k in enumerate(tokens)}
    tokens[' '] = 0
    return tokens


class Tokenizer(object):
    def __init__(self, vocab, max_length=150):
        self.vocab = vocab
        self.max_length = max_length
        self.pad = vocab[' ']

    def c2i(self, c):
        return self.vocab[c]

    def tokenize(self, smile):
        result = []
        for i in smile:
            result.append(self.vocab[i])
        return result
        # return list(map(self.c2i, smile))

    def __call__(self, smile):
        smile = smile
        smile = self.tokenize(smile)
        padded = pad_sequences([smile], maxlen=self.max_length, dtype='int32', padding='pre', truncating='pre',
                               value=self.pad)
        return padded


fin_name = args['data_file']
vocab_name = args['vocab_file']
max_load = args['max_load']
max_length = args['max_length']
test_prob = args['test_size']
print(args)

vocab = load_vocab(vocab_name)
print(len(vocab))
tokenizer = Tokenizer(vocab, max_length=max_length)

# df = pd.read_csv(fin_name, nrows=max_load)
# train_index, test_index = train_test_split(list(range(df.shape[0])), test_size=0.2)
# df_train, df_test = df.iloc[train_index], df.iloc[test_index]

train_toks = []
train_ys = []
test_toks = []
test_ys = []

with open(fin_name, 'r') as fin:
    # skip header
    next(fin)

    for i, line in tqdm(enumerate(fin), total=max_load):
        line = line.split(",")
        smile = line[1]
        result = tokenizer(smile)
        y = float(line[2][:-1])

        if random.random() < test_prob:
            test_toks.append(result)
            test_ys.append(y)
        else:
            train_toks.append(result)
            train_ys.append(y)
        if i > max_load:
            break

X_train = np.concatenate(train_toks, axis=0).astype(np.int32)
y_train = np.array(train_ys, dtype=np.float32)
X_test = np.concatenate(test_toks, axis=0).astype(np.int32)
y_test = np.array(test_ys, dtype=np.float32)

print("%d bytes" % (X_train.size * X_train.itemsize))
print(X_train.shape, X_test.shape)
np.savez_compressed("cleaned.npz", x_test=X_test, x_train=X_train, y_test=y_test, y_train=y_train)
