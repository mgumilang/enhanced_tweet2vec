import sys
from collections import Counter

train_data = sys.argv[1]

min_tweets = 10000

train_x = []
train_y = []

# Preparing data
with open(train_data, 'r') as f:
	lines = f.readlines()
	for line in lines:
		train_split = line.split('\t')
		train_y.append(train_split[0])

labels_list = []

for hashtags in train_y:
	labels = hashtags.split()
	labels_list.extend(labels)

n_label = 0

counts = Counter(labels_list)
for label in counts:
	if counts[label] >= min_tweets:
		n_label += 1
print("Found %s hashtags greater than or equal to %s" % n_label, min_tweets)

x_temp = []
y_temp = []

for idx, line in enumerate(lines):
	label_temp = []
	for label in train_y[idx].split():
		if counts[label] >= min_tweets:
			label_temp.append(label)
	if label_temp:
		x_temp.append(line.split('\t')[1])
		y_temp.append(label_temp)

with open('trimmed.txt', 'w') as f:
	for i in range(len(x_temp)):
		f.write('%s\t%s\n' % y_temp[i], x_temp[i])