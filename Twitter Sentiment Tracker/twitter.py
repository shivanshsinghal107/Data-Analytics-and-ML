import json
import sentiment as s
from keys import *
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener


class listener(StreamListener):
    def on_data(self, data):
        all_data = json.loads(data)
        tweet = all_data['text']
        sentiment, confidence = s.what_is_sentiment(tweet)
        print(f"{tweet} is {sentiment} with a {confidence*100}% confidence")

        if confidence * 100 >= 80:
            output = open('twitter-out.txt', 'a')
            output.write(f"{sentiment}\n")
            output.close()
        return True

    def on_error(self, status):
        print(status)

auth = OAuthHandler(api_key, api_secret)
auth.set_access_token(access_token, access_secret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track = ["hathras"])
