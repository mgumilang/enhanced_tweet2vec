from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.models import load_model

import numpy as np
import sys
from datetime import datetime

num_chars = 70

modelfile = sys.argv[1]
data_size, data_model = modelfile.split('_', 1)
data_type = data_model.split('/')[0]
data_train = data_size + "_data_train.txt"
data_test = data_size + "_data_test.txt"

x_test = []
y_test = []
y_train = []

# Preparing data
with open(data_train, 'r') as f:
	lines = f.readlines()
	for line in lines:
		line = line.split('\t')
		train_label = line[0]
		y_train.append(train_label)

with open(data_test, 'r') as f:
	lines = f.readlines()
	for line in lines:
		line = line.split('\t')
		test_label = line[0]
		test_text = line[1]
		x_test.append(test_text)
		y_test.append(test_label)

if data_type in ('cnn_bi_lstm', 'cnn_bi_gru'):
	# Building char dictionary from x_test
	tk_char = Tokenizer(filters='', char_level=True)
	tk_char.fit_on_texts(x_test)
	char_dict_len = len(tk_char.word_index)
	print("Char dict length = %s" % char_dict_len)

	print("Converting x_test to one-hot vectors..")
	x_test_ohv = []
	x_len = len(x_test)
	i = 1
	for x in x_test:
		if (i % 1000 == 0) or (i == x_len): print("%s of %s" % (i, x_len))
		i += 1
		x_test_ohv.append(sequence.pad_sequences(tk_char.texts_to_matrix(x), maxlen=num_chars, padding='post', truncating='post'))
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
elif data_type == 'bi_gru':
	# Building char dictionary from x_test
	tk_char = Tokenizer(filters='', char_level=True)
	tk_char.fit_on_texts(x_test)
	char_dict_len = len(tk_char.word_index)
	print("Char dict length =", char_dict_len)

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
	print("Word dict length =", word_dict_len)

	print("Converting y_train to vector of class..")
	y_train_v = tk_word.texts_to_matrix(y_train)
	print("Add padding to emit 0 in front of each vector..")
	y_train_v = sequence.pad_sequences(y_train_v, maxlen=word_dict_len)

	print("Converting y_test to vector of class..")
	y_test_v = tk_word.texts_to_matrix(y_test)
	print("Add padding to emit 0 in front of each vector..")
	y_test_v = sequence.pad_sequences(y_test_v, maxlen=word_dict_len)
elif data_type == 'word_cnn_bi_gru':
	# Building word dictionary from x_test
	tk_train = Tokenizer(num_words=maxlen)
	tk_train.fit_on_texts(x_test)
	train_dict_len = len(tk_train.word_index)
	print("Train word dict length =", train_dict_len)

	print("Converting x_test words to integers..")
	x_test_ohv = tk_train.texts_to_sequences(x_train)
	x_len = len(x_test)
	print("Add padding to make {}*word_dict_len matrix..".format(inlen))
	x_test_ohv = sequence.pad_sequences(x_test_ohv, maxlen=inlen, padding='post', truncating='post')

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
else: # data_type == word_bi_gri
	# Building char dictionary from x_test
	tk_char = Tokenizer(filters='', char_level=True)
	tk_char.fit_on_texts(x_test)
	char_dict_len = len(tk_char.word_index)
	print("Char dict length =", char_dict_len)

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
	print("Word dict length =", word_dict_len)

	print("Converting y_train to vector of class..")
	y_train_v = tk_word.texts_to_matrix(y_train)
	print("Add padding to emit 0 in front of each vector..")
	y_train_v = sequence.pad_sequences(y_train_v, maxlen=word_dict_len)

	print("Converting y_test to vector of class..")
	y_test_v = tk_word.texts_to_matrix(y_test)
	print("Add padding to emit 0 in front of each vector..")
	y_test_v = sequence.pad_sequences(y_test_v, maxlen=word_dict_len)

print("Loading model..")
model = load_model(modelfile)
print("Predicting..")
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

print("Calculating evaluation for %s" % modelfile)
print("Recall    @10 = {:3.2f}%".format(recall10(y_test_v, preds) * 100))
print("Precision @1  = {:3.2f}%".format(precision1(y_test_v, preds) * 100))
print("Mean Rank     = {}".format(mean_rank(y_test_v, preds)))

d = datetime.now()
filename = data_size + '_' + data_type + '/' + str(d.date()) + '_' + str(d.hour) + '-' + str(d.minute) + '-' + str(d.second) + '_result.tsv'
hashtag_index = {}
for x in tk_word.word_index:
	hashtag_index[tk_word.word_index[x]] = x

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