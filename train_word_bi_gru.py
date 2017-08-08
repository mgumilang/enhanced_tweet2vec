from __future__ import print_function

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import GRU, Bidirectional
from keras.callbacks import ModelCheckpoint
from keras import optimizers, regularizers

import numpy as np
import os
import errno
import sys

from datetime import datetime

batch_size = 18
epochs = 20
maxlen = 20000
inlen = 80
optimizer = 'RMSprop'

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

# Building word dictionary from x_train
tk_train = Tokenizer(num_words=maxlen)
tk_train.fit_on_texts(x_train)
train_dict_len = len(tk_train.word_index)
print("Train word dict length = %s" % train_dict_len)

print("Converting x_train words to integers..")
x_train_ohv = tk_train.texts_to_sequences(x_train)
x_len = len(x_train)
i = 1
print("Add padding to make {}*word_dict_len matrix..".format(inlen))
x_train_ohv = sequence.pad_sequences(x_train_ohv, maxlen=inlen, padding='post', truncating='post')

print("Converting x_test words to integers..")
x_test_ohv = tk_train.texts_to_sequences(x_test)
x_len = len(x_test)
i = 1
print("Add padding to make {}*word_dict_len matrix..".format(inlen))
x_test_ohv = sequence.pad_sequences(x_test_ohv, maxlen=inlen, padding='post', truncating='post')

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
	starts_from = 0
	model = Sequential()

	model.add(Embedding(maxlen, 128, input_length=inlen))

	model.add(Bidirectional(GRU(96, dropout=0.5)))
	model.add(Dense(word_dict_len, activation='softmax'))

	model.compile(optimizer, 'binary_crossentropy', metrics=['categorical_accuracy'])
else:
	print("Loading model")
	starts_from = int(loaded.split('/')[1].split('-')[0]) + 1
	model = load_model(loaded)

# define the checkpoint
filepath="%s_word_bi_gru/{epoch:02d}-{loss:.4f}.hdf5" % data_size
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
          callbacks=callbacks_list,
          initial_epoch=starts_from,
          validation_split=0.01)

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

print("Evaluation for word Bi-GRU w/ {} optimizer..".format(optimizer))
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

filename = data_size + '_word_bi_gru/' + str(d.date()) + '_' + str(d.hour) + '-' + str(d.minute) + '-' + str(d.second) + '_result.tsv'
modelname = data_size + '_word_bi_gru/' + str(d.date()) + '_' + str(d.hour) + '-' + str(d.minute) + '-' + str(d.second) + '_model.h5'

# Open and write to result_cnn_bi_lstm.tsv
print("Writing result..")
with open(filename, 'w') as f:
	f.write("Tweet\tTrue\tPredicted\n")
	for i in range(len(y_test)):
		predicted = []
		predsort = np.argsort(preds[i])[::-1]
		for idx in predsort[:len(y_test[i].split())]:
			predicted.append("#{}".format(hashtag_index[idx+1]))
		f.write("{}\t{}\t{}\n".format(x_test[i].strip(), y_test[i], ' '.join(predicted)))

print("Saving model to '{}'..".format(modelname))
model.save(modelname)