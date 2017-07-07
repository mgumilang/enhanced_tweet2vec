from __future__ import print_function

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import LSTM, Bidirectional

import numpy as np
import sys

batch_size = 32
epochs = 20

data_train = sys.argv[1]
data_test = sys.argv[2]

x_train = []
y_train = []
x_test = []
y_test = []

# Preparing data
with open(data_train, 'r') as f:
	lines = f.readlines()
	for line in lines:
		line = line.split('\t')
		train_label = line[0]
		train_text = line[1]
		x_train.append(train_text)
		y_train.append(train_label)

with open(data_test, 'r') as f:
	lines = f.readlines()
	for line in lines:
		line = line.split('\t')
		test_label = line[0]
		test_text = line[1]
		x_test.append(test_text)
		y_test.append(test_label)

# Building char dictionary from x_train
tk_char = Tokenizer(filters='', char_level=True)
tk_char.fit_on_texts(x_train)
char_dict_len = len(tk_char.word_index)
print("Char dict length =", char_dict_len)

print("Converting x_train to one-hot vectors..")
x_train_ohv = []
x_len = len(x_train)
i = 1
for x in x_train:
	if (i % 1000 == 0) or (i == x_len): print("%s of %s" % (i, x_len))
	i += 1
	x_train_ohv.append(tk_char.texts_to_matrix(x))
print("Add padding to make 150*char_dict_len matrix..")
x_train_ohv = sequence.pad_sequences(x_train_ohv, maxlen=150, padding='post', truncating='post')

print("Converting x_test to one-hot vectors..")
x_test_ohv = []
x_len = len(x_test)
i = 1
for x in x_test:
	if (i % 1000 == 0) or (i == x_len): print("%s of %s" % (i, x_len))
	i += 1
	x_test_ohv.append(tk_char.texts_to_matrix(x))
print("Add padding to make 150*char_dict_len matrix..")
x_test_ohv = sequence.pad_sequences(x_test_ohv, maxlen=150, padding='post', truncating='post')

# Building word dictionary from y_train
tk_word = Tokenizer()
tk_word.fit_on_texts(y_train)
word_dict_len = len(tk_word.word_index)
print("Word dict length =", word_dict_len)

print("Converting y_train to vector of class..")
y_train_v = tk_word.texts_to_matrix(y_train)
print("Add padding to emit 0 in front of each vector..")
y_train_v = sequence.pad_sequences(y_train_v, maxlen=word_dict_len)

print("Converting y_test to vector of class..")
y_test_v = tk_word.texts_to_matrix(y_test)
print("Add padding to emit 0 in front of each vector..")
y_test_v = sequence.pad_sequences(y_test_v, maxlen=word_dict_len)

print("Building model")
model = Sequential()

model.add(Conv1D(250,
				 3,
				 padding='valid',
				 activation='relu',
				 strides=1,
				 input_shape=(150, char_dict_len+1)))
model.add(MaxPooling1D())
model.add(Dropout(0.25))

model.add(Conv1D(100,
				 2,
				 padding='valid',
				 activation='relu',
				 strides=1))
model.add(MaxPooling1D())
model.add(Dropout(0.25))

model.add(Bidirectional(LSTM(64, dropout=0.5)))
model.add(Dense(word_dict_len, activation='softmax'))

model.compile('adam', 'binary_crossentropy', metrics=['categorical_accuracy'])

print('Train...')
model.fit(x_train_ohv, y_train_v,
          batch_size=batch_size,
          epochs=epochs)

preds = model.predict(x_test_ohv)

sum_precision = 0.
sum_recall = 0.
for i in range(len(y_test_v)):
	idx_preds = list(preds[i])
	idx_preds = np.argsort(idx_preds)
	n_hashtag = sum(y_test_v[i])
	idx_preds = idx_preds[:-(n_hashtag+1):-1]
	pred_hashtag = np.asarray(preds[i])
	pred_hashtag[idx_preds] = 1
	pred_hashtag[pred_hashtag<1] = 0
	sum_hit = sum([a*b for a,b in zip(pred_hashtag, y_test_v[i])])
	recall = float(sum_hit) / n_hashtag
	sum_recall += recall
	precision = float(sum_hit) / n_hashtag
	sum_precision += precision

final_recall = sum_recall / len(y_test_v)
final_precision = sum_precision / len(y_test_v)
f1 = (2 * final_precision * final_recall) / (final_precision + final_recall)
print("Recall    = {:.4f}".format(final_recall))
print("Precision = {:.4f}".format(final_precision))
print("F-score   = {}".format(f1))

# Write result (tweet, true, predicted) to result_cnn_bi_lstm.tsv
# Swap keys and values of tk_word.word_index
hashtag_index = {}
for x in tk_word.word_index:
	hashtag_index[tk_word.word_index[x]] = x
print(hashtag_index)

# Open and write to result_cnn_bi_lstm.tsv
with open('result_cnn_bi_lstm.tsv', 'w') as f:
	for i in range(len(y_test)):
		predicted = []
		for idx, x in enumerate(preds[i]):
			if x == 1:
				predicted.append('#{}'.format(hashtag_index[idx+1]))
		f.write("{}\t{}\t{}\n".format(x_test[i].strip(), y_test[i], ' '.join(predicted)))