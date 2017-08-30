import tweepy
import json
from tweepy import OAuthHandler
import sys
import signal

# Checkpoint init
cp = 0
time = ""
max_id = None

hashtags = []

with open('hashlist.txt', 'r') as f:
	lines = f.readlines()
	for line in lines:
		hashtags.append(line.split()[0])

# Read since_id for continuing fetching tweets
try:
    cont = sys.argv[1]
except:
    cont = None

if cont:
	with open('last_tweet.txt', 'r') as f:
		lines = f.readlines()
		try:
			last = lines[-1]
			cont_hashtag, cont_id = last.split()
			hashtags = hashtags[hashtags.index(cont_hashtag):]
			max_id = cont_id
			print("Continuing: %s %s\n" % (cont_hashtag, cont_id))
		except:
			print("Trouble reading checkpoint..")

consumer_key = 'qJTkb5wFWiSCLLuLWMx0B0Ghl'
consumer_secret = 'Dgwo6GQRWFH3CIwHaUiesfgJw4AloNkZIKXrZFgMcEBQt9kfBr'
access_token = '65892905-16sc7WovrkWGHXrcb1QPfyywr0qvsqeomk4R55WsR'
access_secret = '3u5aO1w1chaEhair50NVbh6BFsBEm68Am1D4RpRA98D4R'

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

# Authorization & wait if twitter API rate limit exceeds
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

# Fetch tweets
with open('custom_unprocessed.txt', 'ab') as f:
	with open('last_tweet.txt', 'w') as g:
		for hashtag in hashtags:
			query = "%s -filter:retweets" % hashtag
			for tweet in tweepy.Cursor(api.search, q=query, max_id=max_id, since="2017-06-27", until="2017-08-31", lang="en").items():
				# Process a single status
				print(str(tweet.created_at), tweet.id, tweet.text)
				f.write(json.dumps(tweet.text.encode('utf8')) + '\n')
				cp = tweet.id
				time = str(tweet.created_at)
				g.write("%s %s\n" % (hashtag, cp))


# Save checkpoint if fetching tweets succeeds
# with open("custom_checkpoint.txt", 'ab') as f:
#     f.write("%s %s kitty\n" % (time, str(cp)))
