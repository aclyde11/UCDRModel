import numpy as np
import pandas as pd
import sys
from modelconfig import args
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
def load_vocab(name):
    tokens = []
    with open(name, 'r') as f:
        for token in f:
            tokens.append(token[:-1])
    tokens = {k : (v+1) for v, k in enumerate(tokens)}
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
        return list(map(self.c2i, smile))

    def __call__(self, smile):
        smile = self.tokenize(smile)
        padded = pad_sequences([smile], maxlen=self.max_length, dtype='int32', padding='pre', truncating='pre', value=self.pad)
        return padded

fin_name = args['data_file']
vocab_name = args['vocab_file']
max_load = args['max_load']
max_length = args['max_length']
print(args)

vocab = load_vocab(vocab_name)
print(len(vocab))
tokenizer = Tokenizer(vocab, max_length=max_length)

df = pd.read_csv(fin_name, nrows=max_load)
train_index, test_index = train_test_split(list(range(df.shape[0])), test_size=0.2)
df_train, df_test = df.iloc[train_index], df.iloc[test_index]

trained_toks = []
ys = []
for i in tqdm(range(df_train.shape[0])):
    trained_toks.append(tokenizer(df_train.iloc[i,1]))
    ys.append(df_train.iloc[i,2])
X_train = np.concatenate(trained_toks, axis=0).astype(np.int32)
y_train = np.array(ys, dtype=np.float32)

trained_toks = []
ys = []
for i in tqdm(range(df_test.shape[0])):
    trained_toks.append(tokenizer(df_test.iloc[i,1]))
    ys.append(df_test.iloc[i,2])
X_test = np.concatenate(trained_toks, axis=0).astype(np.int32)
y_test = np.array(ys, dtype=np.float32)

np.savez_compressed("cleaned.npz", x_test = X_test, x_train = X_train, y_test = y_test, y_train = y_train)