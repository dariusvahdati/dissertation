# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 10:41:38 2023

@author: aliva
"""

import os
import tweepy
import time
import pandas as pd
import re
import nltk
import seaborn as sns
import matplotlib

# Download the NLTK stop words and stemmer
nltk.download("stopwords")
nltk.download("snowball_data")

import logging
logging.basicConfig(filename='app.log', level=logging.DEBUG)

# Use the 'Agg' backend to avoid "Starting a Matplotlib GUI outside of the main thread will likely fail" warning
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, send_from_directory

# Define API keys and access tokens using environment variables
consumer_key = os.environ.get("TWITTER_CONSUMER_KEY")
consumer_secret = os.environ.get("TWITTER_CONSUMER_SECRET")
access_token = os.environ.get("TWITTER_ACCESS_TOKEN")
access_token_secret = os.environ.get("TWITTER_ACCESS_TOKEN_SECRET")

# Authenticate with Twitter API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Define the number of tweets to stream and the number of tweets per page
num_tweets = 50
tweets_per_page = 20

# Define the listener class to handle incoming tweets
class StreamListener(tweepy.StreamListener):
    def __init__(self, max_tweets):
        super().__init__()
        self.tweets = []
        self.max_tweets = max_tweets

    def on_status(self, status):
        if status.retweeted or "RT @" in status.text:
            return True
        if len(self.tweets) < self.max_tweets:
            self.tweets.append(status.text)
            return True
        else:
            return False

    def on_error(self, status_code):
        print(f"Error: {status_code}")
        return False

# Create a function to stream tweets from Twitter
def stream_tweets(keyword):
    # Ensure that the keyword is not None
    if keyword is None:
        return []

    # Create a streaming object and start collecting the tweets
    stream_listener = StreamListener(num_tweets)
    stream = tweepy.Stream(auth=api.auth, listener=stream_listener)
    stream.filter(track=[keyword], languages=["en"])

    # Wait until the desired number of tweets has been collected
    while len(stream_listener.tweets) < num_tweets:
        time.sleep(1)

    # Return the collected tweets
    return stream_listener.tweets


# Create a function to paginate the results
def paginate_results(results, page_num):
    start_index = (page_num - 1) * tweets_per_page
    end_index = page_num * tweets_per_page
    return results[start_index:end_index]



# Load the stop words and stemmer
stop_words = nltk.corpus.stopwords.words("english")
stemmer = nltk.stem.SnowballStemmer("english")

# Create a function to preprocess the tweet text
def preprocess_tweet_text(text):
    # Remove URLs and mentions
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\S+", "", text)

    # Convert to lowercase and split into words
    words = text.lower().split()

    # Remove stop words and perform stemming
    words = [stemmer.stem(word) for word in words if word not in stop_words]

    # Join the words back into a string and return
    return " ".join(words)

# Create a function to store the tweet data in a Pandas dataframe
def store_tweet_data(tweets):
    # Preprocess the tweet text
    tweets = [preprocess_tweet_text(tweet) for tweet in tweets]

    # Store the tweet data in a Pandas dataframe
    df = pd.DataFrame({"text": tweets})

    return df

def perform_eda(df):
    num_tweets = len(df)
    df["text"] = df["text"].astype(str)
    
    avg_tweet_len = df["text"].str.len().mean()
    top_words = pd.Series(" ".join(df["text"]).lower().split()).value_counts()[:5]

    # Create a bar chart of the top 5 most frequent words
    if not top_words.empty:
        plt.figure(figsize=(7, 5))
        plt.title("Top 5 most frequent words")
        sns.barplot(x=top_words.index, y=top_words.values)
        plt.xlabel("Word")
        plt.ylabel("Frequency")
        plt.tight_layout()

        # Save the plot to a file
        plot_filename = "top_words.png"
        plt.savefig(os.path.join("static", plot_filename))

        # Return the EDA results and the filename of the plot
        return num_tweets, avg_tweet_len, top_words, plot_filename
    else:
        return num_tweets, avg_tweet_len, top_words, None


# Create a Flask app
app = Flask(__name__)
app.config["NUM_TWEETS"] = num_tweets

# Define a route to display the form
@app.route("/")
def index():
    return render_template("form.html")

# Add a new route to serve static files
@app.route("/static/<path:filename>")
def serve_static(filename):
    root_dir = os.getcwd()
    return send_from_directory(os.path.join(root_dir, "static"), filename)


# Define a route to handle the form submission
@app.route("/submit", methods=["GET", "POST"])
def submit():
    # Retrieve the keyword and page number entered by the user
    if request.method == "POST":
        keyword = request.form.get('keyword')
        page_num = int(request.form.get("page_num", 1))
    else:
        keyword = request.args.get('keyword')
        page_num = int(request.args.get("page_num"))

    # Stream live tweets from Twitter
    tweets = stream_tweets(keyword)

    # Store the tweet data in a Pandas dataframe
    df = store_tweet_data(tweets)

    # Paginate the results
    results = paginate_results(df["text"].tolist(), page_num)
    num_pages = int(len(df) / tweets_per_page) + 1

    # Perform EDA on the data
    num_tweets, avg_tweet_len, top_words, plot_filename = perform_eda(df)

    # Render the template with the results
    return render_template("results.html", keyword=keyword, results=results, num_pages=num_pages, page_num=page_num, num_tweets=num_tweets, avg_tweet_len=avg_tweet_len, top_words=top_words, plot_filename=plot_filename)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)