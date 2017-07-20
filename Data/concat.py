import sys

text1 = sys.argv[1]
text2 = sys.argv[2]
out = sys.argv[3]

lines = []
with open(text1, 'r') as f:
	lines = f.readlines()

with open(text2, 'r') as f:
	lines.extend(f.readlines())

with open(out, 'w') as f:
	for line in lines:
		f.write(line)