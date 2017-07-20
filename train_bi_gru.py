from __future__ import print_function

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import GRU, Bidirectional
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras import regularizers

import numpy as np
import os
import errno
import sys

from datetime import datetime

batch_size = 32
epochs = 20
optimizer = 'Adam'

data_size = sys.argv[1]
try:
	loaded = sys.argv[2]
except:
	loaded = None
data_train = data_size + "_data_train.txt"
data_test = data_size + "_data_test.txt"

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
print("Char dict length = %s" % char_dict_len)

print("Converting x_train chars to integers..")
x_train_ohv = []
x_len = len(x_train)
i = 1
for x in x_train:
	if (i % 1000 == 0) or (i == x_len): print("%s of %s" % (i, x_len))
	i += 1
	x_temp = [tk_char.word_index[c] for c in x]
	x_train_ohv.append(x_temp)
print("Add padding to make 150*char_dict_len matrix..")
x_train_ohv = sequence.pad_sequences(x_train_ohv, maxlen=150, padding='post', truncating='post')

print("Converting x_test chars to integers..")
x_test_ohv = []
x_len = len(x_test)
i = 1
for x in x_test:
	if (i % 1000 == 0) or (i == x_len): print("%s of %s" % (i, x_len))
	i += 1
	x_temp = [tk_char.word_index[c] for c in x]
	x_test_ohv.append(x_temp)
print("Add padding to make 150*char_dict_len matrix..")
x_test_ohv = sequence.pad_sequences(x_test_ohv, maxlen=150, padding='post', truncating='post')

# Building word dictionary from y_train
tk_word = Tokenizer()
tk_word.fit_on_texts(y_train)
word_dict_len = len(tk_word.word_index)
print("Word dict length = %s" % word_dict_len)

print("Converting y_train to vector of class..")
y_train_v = tk_word.texts_to_matrix(y_train)
print("Add padding to emit 0 in front of each vector..")
y_train_v = sequence.pad_sequences(y_train_v, maxlen=word_dict_len)

print("Converting y_test to vector of class..")
y_test_v = tk_word.texts_to_matrix(y_test)
print("Add padding to emit 0 in front of each vector..")
y_test_v = sequence.pad_sequences(y_test_v, maxlen=word_dict_len)

if not loaded:
	print("Building model")
	model = Sequential()

	model.add(Embedding(70, 128, input_length=150))

	model.add(Bidirectional(GRU(64, dropout=0.5)))
	model.add(Dense(word_dict_len, activation='softmax'))

	model.compile(optimizer, 'binary_crossentropy', metrics=['categorical_accuracy'])
else:
	print("Loading model")
	model = load_model(loaded)

# define the checkpoint
filepath="%s_bi_gru/{epoch:02d}-{loss:.4f}.hdf5" % data_size
if not os.path.exists(os.path.dirname(filepath)):
    try:
        os.makedirs(os.path.dirname(filepath))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

print('Train...')
model.fit(x_train_ohv, y_train_v,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=callbacks_list)

preds = model.predict(x_test_ohv)

def precision1(y_true, y_pred):
	sum_precision = 0.
	for i in range(len(y_true)):
		y_pred_idx = np.argsort(y_pred[i])
		if y_true[i][y_pred_idx[-1]]:
			sum_precision += 1
	return sum_precision/len(y_true)

def recall10(y_true, y_pred):
	sum_recall = 0.
	for i in range(len(y_true)):
		y_pred_idx = np.argsort(y_pred[i])[:-11:-1]
		sum_hit = 0.
		for j in y_pred_idx:
			if (y_pred[i][j]) and (y_true[i][j]):
				sum_hit += 1
		sum_recall += sum_hit / sum(y_true[i])
	return sum_recall/len(y_true)

def mean_rank(y_true, y_pred):
	sum_rank = 0
	for i in range(len(y_true)):
		y_pred_idx = np.argsort(y_pred[i])[::-1]
		for idx, j in enumerate(y_pred_idx):
			if y_true[i][j]:
				sum_rank += idx
	return sum_rank/np.sum(y_true)

print("Evaluation for Bi-BRU w/ {} optimizer..".format(optimizer))
print("Recall    @10 = {:3.2f}%".format(recall10(y_test_v, preds) * 100))
print("Precision @1  = {:3.2f}%".format(precision1(y_test_v, preds) * 100))
print("Mean Rank     = {}".format(mean_rank(y_test_v, preds)))

# Write result (tweet, true, predicted) to result_cnn_bi_lstm.tsv
# Swap keys and values of tk_word.word_index
hashtag_index = {}
for x in tk_word.word_index:
	hashtag_index[tk_word.word_index[x]] = x
print(hashtag_index)

d = datetime.now()

filename = data_size + '_bi_GRU/' + str(d.date()) + '_' + str(d.hour) + '-' + str(d.minute) + '-' + str(d.second) + '_result.tsv'
modelname = data_size + '_bi_GRU/' + str(d.date()) + '_' + str(d.hour) + '-' + str(d.minute) + '-' + str(d.second) + '_model.h5'

# Open and write to result_cnn_bi_lstm.tsv
print("Writing result..")
with open(filename, 'w') as f:
	for i in range(len(y_test)):
		predicted = []
		for idx, x in enumerate(preds[i]):
			if x == 1:
				predicted.append('#{}'.format(hashtag_index[idx+1]))
		f.write("{}\t{}\t{}\n".format(x_test[i].strip(), y_test[i], ' '.join(predicted)))

print("Saving model..")
model.save(modelname)