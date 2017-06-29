import re
import sys
import io
import json

# input and output files
infile = sys.argv[1]
outfile = sys.argv[2]

regex_str = [
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs

    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)+' # anything else
]

tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)

def tokenize(s):
    return tokens_re.findall(s)

def preprocess(s, lowercase=True):
    tokens = tokenize(s)
    tokens = [token.lower() for token in tokens]

    html_regex = re.compile('<[^>]+>')
    tokens = [token for token in tokens if not html_regex.match(token)]

    mention_regex = re.compile('(?:@[\w_]+)')
    tokens = ['@user' if mention_regex.match(token) else token for token in tokens]

    url_regex = re.compile('http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+')
    tokens = ['!url' if url_regex.match(token) else token for token in tokens]

    tokens = ['' if hashtag_regex.match(token) else token for token in tokens]

    flag = False
    for item in tokens:
        if item=='rt':
            flag = True
            continue
        if flag and item=='@user':
            return ''
        else:
            flag = False

    return ' '.join([t for t in tokens if t]).replace('rt @user : ','')

hashtag_regex = re.compile("(?:\#+[\w_]+[\w\'_\-]*[\w_]+)")

with open(infile, 'r') as f:
    res = []
    lines = f.readlines()
    for line in lines:
        line = line[1:-2]
        line = line.replace('\\n', '')
        if (hashtag_regex.search(line)):
            hashtag = re.findall(r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", line)
            tweet_processed_text = preprocess(line)
            data = ' '.join(hashtag) + '\t' + tweet_processed_text
            res.append(data)

with open(outfile, 'w') as f:
    for text in res:
        f.write(text + '\n')
