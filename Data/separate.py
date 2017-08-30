import sys
import operator
import random
import numpy as np

from collections import OrderedDict
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

train_data = sys.argv[1]

x_train = []
y_train = []
x_test = []
y_test = []
x_temp = []
y_temp = []
x_temp1 = []
y_temp1 = []

test_num = 50000
label_min_threshold = 500
label_max_threshold = 20000

# Preparing data
with open(train_data, 'r') as f:
	lines = f.readlines()
	random.shuffle(lines)
	for line in lines:
		line = line.split('\t')
		train_label = line[0]
		train_label = train_label.split(' ')
		train_text = line[1]
		x_train.append(train_text)
		y_train.append(train_label)

y_train = [[y.lower() for y in labels] for labels in y_train]

# Calculating hashtag's frequency and write hashtags to file
hashcount = OrderedDict()
for ll in y_train:
	for y in ll:
		if y not in hashcount:
			hashcount[y] = 0
		hashcount[y] += 1
sorted_hash = sorted(hashcount.items(), key=operator.itemgetter(1))
with open("hashtags_raw.txt", 'w') as f:
	f.write('\n'.join('%s %s' % x for x in sorted_hash))

print("Eliminate unnecessary hashtags")
for i, labels in enumerate(y_train):
	label_temp = []
	for label in labels:
		if (hashcount[label] >= label_min_threshold) and (hashcount[label] <= label_max_threshold):
			label_temp.append(label)
	if label_temp:
		x_temp.append(x_train[i])
		y_temp.append(label_temp)

# Calculating hashtag's frequency and write hashtags to file
hashcount = OrderedDict()
for ll in y_temp:
	for y in ll:
		if y not in hashcount:
			hashcount[y] = 0
		hashcount[y] += 1
sorted_hash = sorted(hashcount.items(), key=operator.itemgetter(1))
with open("hashtags.txt", 'w') as f:
	f.write('\n'.join('%s %s' % x for x in sorted_hash))

count = 0
for x in hashcount.keys():
	if hashcount[x] >= label_min_threshold:
		count += 1
print("hashtags count =", count)

print("Final    = {:5d}".format(len(x_temp)))

x_test, y_test = [], []
x_train, y_train = [], []

# Split into training and testing data
# Making sure each hashtag is in training data
found = False
labels = hashcount.keys()
for label in labels:
	if hashcount[label] >= label_min_threshold:
		found = False
		for train_labels in y_train:
			if label in train_labels:
				found = True
				break
		if not found:		
			for i, y in enumerate(y_temp):
				if label in y:
					x_train.append(x_temp[i])
					y_train.append(y_temp[i])
					del x_temp[i]
					del y_temp[i]
					break

x_test.extend(x_temp[:test_num])
y_test.extend(y_temp[:test_num])
x_train.extend(x_temp[test_num:])
y_train.extend(y_temp[test_num:])

with open('data_train.txt', 'w') as f:
	for i in range(0, len(x_train)):
		labels = ' '.join(y_train[i])
		f.write("{}\t{}".format(labels, x_train[i]))
with open('data_test.txt', 'w') as f:
	for i in range(0, len(x_test)):
		labels = ' '.join(y_test[i])
		f.write("{}\t{}".format(labels, x_test[i]))
