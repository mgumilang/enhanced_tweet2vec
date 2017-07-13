import tweepy
import json
from tweepy import OAuthHandler
import sys
import signal

# Checkpoint init
cp = 0
time = ""

# Read since_id for continuing fetching tweets
try:
    max_id = int(sys.argv[1])
except:
    max_id = None

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
    for tweet in tweepy.Cursor(api.search, q="#exo -filter:retweets", max_id=max_id, since="2017-06-27", until="2017-07-13", lang="en").items():
        # Process a single status
        print(str(tweet.created_at), tweet.id, tweet.text)
        f.write(json.dumps(tweet.text.encode('utf8')) + '\n')
        cp = tweet.id
        time = str(tweet.created_at)

# Save checkpoint if fetching tweets succeeds
with open("custom_checkpoint.txt", 'ab') as f:
    f.write("%s %s exo\n" % (time, str(cp)))
