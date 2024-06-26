"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import requests
import numpy as np
import segutil
import itertools
import tiktoken
__file__='/Users/rastislavhronsky/ML-research/nanoGPT/data/shakespeare/prepare.py'
# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

train_size = .9
data_train = data[:int(len(data)*train_size)]
data_test = data[int(len(data)*train_size):]
gptenc = tiktoken.get_encoding("gpt2")
tiktoken_train = gptenc.encode_ordinary(data_train)
tiktoken_train = gptenc.decode_batch([[el] for el in tiktoken_train])
tiktoken_test = gptenc.encode_ordinary(data_test)
tiktoken_test = gptenc.decode_batch([[el] for el in tiktoken_test])
tiktoken_seg_train = [(i, len(w)) for i, w in enumerate(tiktoken_train)]
tiktoken_seg_test = [(i, len(w)) for i, w in enumerate(tiktoken_test)]
char_seg_train = [(i, 1) for i in range(len(data_train))]
char_seg_test = [(i, 1) for i in range(len(data_test))]
cps_train = segutil.Corpus(data_train, segmentation={'char': char_seg_train, 'tiktoken': tiktoken_seg_train,},)
cps_test = segutil.Corpus(data_test, segmentation={'char': char_seg_test, 'tiktoken': tiktoken_seg_test,},)
cps_train.save('cps_train.pkl')
cps_test.save('cps_test.pkl')
vocab_char = segutil.build_vocab(data_train, sort_by='alpha', unk_token=None)
print('char vocab size: ', len(vocab_char))

gpt_types = gptenc.decode_tokens_bytes(list(range(gptenc.n_vocab)))
gpt_vocab = segutil.build_vocab(gpt_types, sort_by='dont', unk_token=None)
print('tiktoken vocab size: ', len(gpt_vocab))
vocab_char.save('vocab_char.pkl')
gpt_vocab.save('vocab_gpt.pkl')