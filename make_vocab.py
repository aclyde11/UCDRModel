import pandas as pd
import sys
from tqdm import tqdm

fin_name = './data/AmpC_screen_table.csv'
fout_name = "vocab.txt"
skip_header = True

vocab = set()
with open(fin_name, 'r') as fin:
    _ = next(fin)
    for line in tqdm(fin):
        smile = line.split(',')[1]
        vocab.update(smile)

with open(fout_name, 'w') as fout:
    for token in (list(vocab)):
        fout.write(token)
        fout.write('\n')

print(vocab)
print(len(vocab))