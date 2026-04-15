import tweepy

def get_tweets(api_key, api_secret, access_token, access_secret, query="AI", count=10):
    auth = tweepy.OAuth1UserHandler(api_key, api_secret, access_token, access_secret)
    api = tweepy.API(auth)

    tweets = api.search_tweets(q=query, count=count, lang="en")
    return [tweet.text for tweet in tweets]