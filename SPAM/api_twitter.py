import re

import tweepy
import yaml


def authorize():
    with open("custom_properties.yml", 'r') as yaml_file:
        properties = yaml.load(yaml_file)

    twitter_consumer_key = properties['twitter']['consumer_key']
    twitter_consumer_secret = properties['twitter']['consumer_secret']
    twitter_access_token = properties['twitter']['access_token']
    twitter_access_token_secret = properties['twitter']['access_token_secret']

    auth = tweepy.OAuthHandler(twitter_consumer_key, twitter_consumer_secret)
    auth.set_access_token(twitter_access_token, twitter_access_token_secret)
    api = tweepy.API(auth)
    return api


def get_tweets(username, count):
    api = authorize()
    tweets = api.user_timeline(screen_name=username, count=count)

    tweets_status = []
    for tweet in tweets:
        tweets_status.append(remove_emoticons(tweet.text))

    return tweets_status


def remove_emoticons(text):
    pattern = re.compile('['
                         u'\U0001F600-\U0001F64F'  # emoticons
                         u'\U0001F300-\U0001F5FF'  # symbols & pictographs
                         u'\U0001F680-\U0001F6FF'  # transport & map symbols
                         u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
                         ']+', flags=re.UNICODE)
    text = pattern.sub(r'', text)
    return text

