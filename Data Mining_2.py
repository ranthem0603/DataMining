
# Part 3: Text mining.

# Return a pandas dataframe containing the data set.
# Specify a 'latin-1' encoding when reading the data.
# data_file will be populated with a string 
# corresponding to a path containing the wholesale_customers.csv file.
#%%
import pandas as pd
import re
from collections import Counter
import requests
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
#%%
def read_csv_3(data_file):
	df = pd.read_csv(data_file, encoding='latin1')
	return df
#%%
# Return a list with the possible sentiments that a tweet might have.
def get_sentiments(df):
    sentiments = df['Sentiment'].unique().tolist()
    return sentiments
#%%
# Return a string containing the second most popular sentiment among the tweets.
def second_most_popular_sentiment(df):
    sentiment_counts = df['Sentiment'].value_counts()
    second_most_popular_sentiment = sentiment_counts.index[1]
    return str(second_most_popular_sentiment) 
#%%

# Return the date (string as it appears in the data) with the greatest number of extremely positive tweets.
def date_most_popular_tweets(df):
    extremely_positive_dates = df[df['Sentiment'] == 'Extremely Positive']['TweetAt'].value_counts()
    date_with_most_extremely_positive_tweets = extremely_positive_dates.idxmax()
    return date_with_most_extremely_positive_tweets
#%%
# Modify the dataframe df by converting all tweets to lower case. 
def lower_case(df):
    df['OriginalTweet'] = df['OriginalTweet'].apply(lambda x: x.lower())
    return df
#%%
# Modify the dataframe df by replacing each characters which is not alphabetic or whitespace with a whitespace.
def remove_non_alphabetic_chars(df):
    df['OriginalTweet'] = df['OriginalTweet'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', ' ', x))
    return df
#%%

# Modify the dataframe df with tweets after removing characters which are not alphabetic or whitespaces.
def remove_multiple_consecutive_whitespaces(df):
    df['OriginalTweet'] = df['OriginalTweet'].apply(lambda x: re.sub(r'\s+', ' ', x))
    return df

#%%
# Given a dataframe where each tweet is one string with words separated by single whitespaces,
# tokenize every tweet by converting it into a list of words (strings).
def tokenize(df):
    df['tokenized_tweets'] = df['OriginalTweet'].apply(lambda x: [str(word) for word in x.split()])
    return df['tokenized_tweets']
#%%

# Given dataframe tdf with the tweets tokenized, return the number of words in all tweets including repetitions.
def count_words_with_repetitions(tdf):
    total_words = 0
    for tweet in tdf:
        total_words += len(tweet)
    return total_words
#%%
# Given dataframe tdf with the tweets tokenized, return the number of distinct words in all tweets.
def count_words_without_repetitions(tdf):     
    unique_words = set()
    for tweet in tdf:
        unique_words.update(tweet)
    return len(unique_words)
#%%

# Given dataframe tdf with the tweets tokenized, return a list with the k distinct words that are most frequent in the tweets.
def frequent_words(tdf,k):
    word_counter = Counter()
    for tweet in tdf:
        word_counter.update(tweet)
    most_common_words = word_counter.most_common(k)
    return [word for word, _ in most_common_words]
#%%

# Given dataframe tdf with the tweets tokenized, remove stop words and words with <=2 characters from each tweet.
# The function should download the list of stop words via:
# https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt
def remove_stop_words(tdf):
    stop_words_url = "https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt"
    response = requests.get(stop_words_url)
    stop_words = set(response.text.split())
    for index, tokenized_tweet in tdf.items():
        if not isinstance(tokenized_tweet, list):
            tokenized_tweet = tokenized_tweet.split()
        filtered_words = [word for word in tokenized_tweet if word not in stop_words and len(word) > 2]
        tdf.at[index] = filtered_words
    return tdf
#%%

# Given dataframe tdf with the tweets tokenized, reduce each word in every tweet to its stem.
def stemming(tdf):
    stemmer = PorterStemmer()
    tdf = tdf.apply(lambda tweet: [stemmer.stem(word) for word in tweet])
    return tdf
#%%

# Given a pandas dataframe df with the original coronavirus_tweets.csv data set,
# build a Multinomial Naive Bayes classifier. 
# Return predicted sentiments (e.g. 'Neutral', 'Positive') for the training set
# as a 1d array (numpy.ndarray). 
def mnb_predict(df):
    tweets = df['OriginalTweet'].values
    labels = df['Sentiment'].values
    X_train, X_test, y_train, y_test = train_test_split(tweets, labels, test_size=0.2, random_state=42)
    vectorizer = CountVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    clf = MultinomialNB()
    clf.fit(X_train_vectorized, y_train)
    X_test_vectorized = vectorizer.transform(X_test)
    return clf.predict(X_test_vectorized)
#%%
# Given a 1d array (numpy.ndarray) y_pred with predicted labels (e.g. 'Neutral', 'Positive') 
# by a classifier and another 1d array y_true with the true labels, 
# return the classification accuracy rounded in the 3rd decimal digit.
def mnb_accuracy(y_pred,y_true):
    correct_predictions = sum(1 for pred, true in zip(y_pred, y_true) if pred == true)
    total_samples = len(y_pred)
    accuracy = correct_predictions / total_samples
    accuracy = round(accuracy, 3)
    return accuracy