from __future__ import print_function

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D

import sys

train_data = sys.argv[1]

x_train = []
y_train = []
x_test = []
y_test = []

# Preparing data
with open(train_data, 'r') as f:
	lines = f.readlines()

	for line in lines:
		line = line.split('\t')
		train_label = line[0]
		train_text = line[1]
		x_train.append(train_text)
		y_train.append(train_label)

# Building char dictionary from x_train
tokenizer = Tokenizer(filters='', char_level=True)
tokenizer.fit_on_texts(x_train)
char_dict_len = len(tokenizer.word_index)
print("Char dict length =", char_dict_len)

x_train_ohv = [tokenizer.texts_to_matrix(x) for x in x_train]
x_train_ohv = sequence.pad_sequences(x_train_ohv, maxlen=150, padding='post', truncating='post')

# Building word dictionary from y_train
tokenizer = Tokenizer()
tokenizer.fit_on_texts(y_train)
word_dict_len = len(tokenizer.word_index)
print("Word dict length =", word_dict_len)
print(tokenizer.word_index)

y_train_v = tokenizer.texts_to_matrix(y_train)
y_train_v = sequence.pad_sequences(y_train_v, maxlen=len(tokenizer.word_index))
