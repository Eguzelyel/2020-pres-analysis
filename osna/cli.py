# -*- coding: utf-8 -*-

"""Console script for elevate_osna."""

# add whatever imports you need.
# be sure to also add to requirements.txt so I can install them.
from collections import Counter
import time
import collections
import click
import json
import glob
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import scipy.stats as scistats

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, EmotionOptions, SentimentOptions, KeywordsOptions
from wordcloud import WordCloud, STOPWORDS

from . import credentials_path
from TwitterAPI import TwitterAPI


@click.group()
def main(args=None):
    """Console script for osna."""
    return 0


@main.command('collect')
@click.argument('directory', type=click.Path(exists=True))
def collect(directory):
    """
    Collect data and store in given directory.

    This should collect any data needed to train and evaluate your approach.
    This may be a long-running job (e.g., maybe you run this command and let it go for a week).
    """
    # Create an instance of the Twitter API
    twitter = get_twitter()

    # Collect the raw twitter data. Due to limits on the API, this will collect ~3000 tweets per candidate
    warren_tweets = get_timeline_for_candidate(twitter, 5000, 'SenWarren')
    biden_tweets = get_timeline_for_candidate(twitter, 5000, 'JoeBiden')
    bernie_tweets = get_timeline_for_candidate(twitter, 5000, 'BernieSanders')
    yang_tweets = get_timeline_for_candidate(twitter, 5000, 'AndrewYang')

    # Create an instance of the IBM NLU tool
    nlu = get_ibm_nlu()

    # Perform sentiment analysis on Warren's older tweets
    warren_tweet_text = [tweet['full_text']
                         for tweet in warren_tweets[1000:]]
    warren_senti_scores = sentiment_analysis(warren_tweet_text, nlu)

    # Filter out the unparsable tweets from warren_tweets
    for tweet_text, score in warren_senti_scores.items():
        if score == 'N/A, could not parse':
            warren_tweets = [
                tweet for tweet in warren_tweets if tweet['full_text'] != tweet_text]

    # Remove the unparsable entries from the sentiment scores dictionary
    warren_senti_scores = {key: val for key, val in warren_senti_scores.items(
    ) if val != 'N/A, could not parse'}

    # Perform sentiment analysis on Biden's older tweets
    biden_tweet_text = [tweet['full_text']
                        for tweet in biden_tweets[1000:]]
    biden_senti_scores = sentiment_analysis(biden_tweet_text, nlu)

    # Filter out the unparsable tweets from biden_tweets
    for tweet_text, score in biden_senti_scores.items():
        if score == 'N/A, could not parse':
            biden_tweets = [
                tweet for tweet in biden_tweets if tweet['full_text'] != tweet_text]

    # Remove the unparsable entries from the sentiment scores dictionary
    biden_senti_scores = {key: val for key, val in biden_senti_scores.items(
    ) if val != 'N/A, could not parse'}

    # Perform sentiment analysis on Bernie's older tweets
    bernie_tweet_text = [tweet['full_text']
                         for tweet in bernie_tweets[1000:]]
    bernie_senti_scores = sentiment_analysis(bernie_tweet_text, nlu)

    # Filter out the unparsable tweets from bernie_tweets
    for tweet_text, score in bernie_senti_scores.items():
        if score == 'N/A, could not parse':
            bernie_tweets = [
                tweet for tweet in bernie_tweets if tweet['full_text'] != tweet_text]

    # Remove the unparsable entries from the sentiment scores dictionary
    bernie_senti_scores = {key: val for key, val in bernie_senti_scores.items(
    ) if val != 'N/A, could not parse'}

    # Perform sentiment analysis on Yang's older tweets
    yang_tweet_text = [tweet['full_text']
                       for tweet in yang_tweets[1000:]]
    yang_senti_scores = sentiment_analysis(yang_tweet_text, nlu)

    # Filter out the unparsable tweets from yang_tweets
    for tweet_text, score in yang_senti_scores.items():
        if score == 'N/A, could not parse':
            yang_tweets = [
                tweet for tweet in yang_tweets if tweet['full_text'] != tweet_text]

    # Remove the unparsable entries from the sentiment scores dictionary
    yang_senti_scores = {key: val for key, val in yang_senti_scores.items(
    ) if val != 'N/A, could not parse'}

    # Save the older tweets of each candidate to their respective pickle files
    pickle.dump(warren_tweets[1000:], open(
        f'{directory}/tweets/old/warren_tweets_old.pkl', 'wb'))
    pickle.dump(biden_tweets[1000:], open(
        f'{directory}/tweets/old/biden_tweets_old.pkl', 'wb'))
    pickle.dump(bernie_tweets[1000:], open(
        f'{directory}/tweets/old/bernie_tweets_old.pkl', 'wb'))
    pickle.dump(yang_tweets[1000:], open(
        f'{directory}/tweets/old/yang_tweets_old.pkl', 'wb'))

    # Save the newer 1000 tweets of each candidate to separate pickle files
    # (for use in testing)
    pickle.dump(warren_tweets[:1000],
                open(f'{directory}/tweets/new/warren_tweets_new_1000.p', 'wb'))
    pickle.dump(biden_tweets[:1000],
                open(f'{directory}/tweets/new/biden_tweets_new_1000.p', 'wb'))
    pickle.dump(bernie_tweets[:1000],
                open(f'{directory}/tweets/new/bernie_tweets_new_1000.p', 'wb'))
    pickle.dump(yang_tweets[:1000],
                open(f'{directory}/tweets/new/yang_tweets_new_1000.p', 'wb'))

    # Pickle the sentiment analysis scores for the tweets of all of the candidates into separate files
    pickle.dump(warren_senti_scores, open(
        f'{directory}/senti_scores/warren_senti_scores.pkl', 'wb'))
    pickle.dump(biden_senti_scores, open(
        f'{directory}/senti_scores/biden_senti_scores.pkl', 'wb'))
    pickle.dump(bernie_senti_scores, open(
        f'{directory}/senti_scores/bernie_senti_scores.pkl', 'wb'))
    pickle.dump(yang_senti_scores, open(
        f'{directory}/senti_scores/yang_senti_scores.pkl', 'wb'))


@main.command('evaluate')
@click.argument('directory', type=click.Path(exists=True))
def evaluate(directory):
    """
    Report accuracy and other metrics of your approach.
    For example, compare classification accuracy for different
    methods. The directory argument refers to where the results are stores.
    In the case of this project, the results are stored in the evaluate folder
    in the data folder at the root of the project, so running "osna evaluate data"
    will return all of the evaluations.
    """
    # Take the outputs of train function.
    # train() simultaneously dumps evaluations while training.
    [warren_train_acc, warren_test_acc, warren_train_f1, warren_test_f1] = pickle.load(
        open(f'{directory}/evaluate/warren_evaluate.pk', 'rb'))
    [biden_train_acc, biden_test_acc, biden_train_f1, biden_test_f1] = pickle.load(
        open(f'{directory}/evaluate/biden_evaluate.pk', 'rb'))
    [bernie_train_acc, bernie_test_acc, bernie_train_f1, bernie_test_f1] = pickle.load(
        open(f'{directory}/evaluate/bernie_evaluate.pk', 'rb'))
    [yang_train_acc, yang_test_acc, yang_train_f1, yang_test_f1] = pickle.load(
        open(f'{directory}/evaluate/yang_evaluate.pk', 'rb'))

    # Display it to the user.
    print("LogisticRegression Classifier Evaluation")
    print("\t", "Train Acc\t", "Test Acc\t", "Train F1 Score\t", "Test F1 Score")
    print("Warren\t", '{:3.4f}'.format(warren_train_acc), "\t", '{:3.4f}'.format(warren_test_acc), "\t",
      '{:3.4f}'.format(warren_train_f1), "\t", '{:3.4f}'.format(warren_test_f1))
    print("Biden\t", '{:3.4f}'.format(biden_train_acc), "\t", '{:3.4f}'.format(biden_test_acc), "\t",
      '{:3.4f}'.format(biden_train_f1), "\t", '{:3.4f}'.format(biden_test_f1))
    print("Bernie\t", '{:3.4f}'.format(bernie_train_acc), "\t", '{:3.4f}'.format(bernie_test_acc), "\t",
      '{:3.4f}'.format(bernie_train_f1), "\t", '{:3.4f}'.format(bernie_test_f1))
    print("Yang\t", '{:3.4f}'.format(yang_train_acc), "\t", '{:3.4f}'.format(yang_test_acc), "\t",
      '{:3.4f}'.format(yang_train_f1), "\t", '{:3.4f}'.format(yang_test_f1))

@main.command('network')
@click.argument('directory', type=click.Path(exists=True))
@click.argument('image_dir', type=click.Path(exists=True))
def network(directory, image_dir):
    """
    Perform the network analysis component of your project.
    E.g., compute network statistics, perform clustering
    or link prediction, etc.

    Once of the network analysis is performed, the resulting plots are stored
    in the directory passed in as image_dir
    """
    # Count the number of retweets for each candidate and plot them in a separate
    # histogram for each of them

    warren_tweets = pickle.load(
        open(f'{directory}/tweets/old/warren_tweets_old.pkl', 'rb'))
    biden_tweets = pickle.load(
        open(f'{directory}/tweets/old/biden_tweets_old.pkl', 'rb'))
    bernie_tweets = pickle.load(
        open(f'{directory}/tweets/old/bernie_tweets_old.pkl', 'rb'))
    yang_tweets = pickle.load(
        open(f'{directory}/tweets/old/yang_tweets_old.pkl', 'rb'))

    # Create the 2x2 grid for plotting the histograms for each candidate
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    # Create Warren plot
    num_retweets_warren = np.array([warren_tweets[i]['retweet_count']
                                    for i in range(len(warren_tweets))])

    # Estimate parameters of inverse gamma dist
    sorted_warren_counts = sorted(num_retweets_warren, reverse=True)
    shape, loc, scale = scistats.invgamma.fit(sorted_warren_counts)
    rv = scistats.invgamma(shape, loc, scale)

    # Freedman–Diaconis rule for histogram bin selection
    iqr = scistats.iqr(sorted_warren_counts)
    n_bins = int((2 * iqr) // np.cbrt(len(sorted_warren_counts)))

    warren_linspace = np.linspace(0, max(sorted_warren_counts))

    axs[0, 0].hist(sorted_warren_counts, bins=n_bins, density=True)
    axs[0, 0].plot(warren_linspace, rv.pdf(warren_linspace))
    axs[0, 0].set_title('Warren Retweet Counts')

    # Create Biden plot
    num_retweets_biden = np.array([biden_tweets[i]['retweet_count']
                                   for i in range(len(biden_tweets))])

    # Estimate parameters of inverse gamma dist
    sorted_biden_counts = sorted(num_retweets_biden, reverse=True)
    shape, loc, scale = scistats.invgamma.fit(sorted_biden_counts)
    rv = scistats.invgamma(shape, loc, scale)

    # Freedman–Diaconis rule for histogram bin selection
    iqr = scistats.iqr(sorted_biden_counts)
    num_bins = int((2 * iqr) // np.cbrt(len(sorted_biden_counts)))

    biden_linspace = np.linspace(0, max(sorted_biden_counts))

    axs[0, 1].hist(sorted_biden_counts, bins=num_bins, density=True)
    axs[0, 1].plot(biden_linspace, rv.pdf(biden_linspace))
    axs[0, 1].set_title('Biden Retweet Counts')

    # Create Bernie plot

    num_retweets_bernie = np.array([bernie_tweets[i]['retweet_count']
                                    for i in range(len(bernie_tweets))])

    # Estimate parameters of inverse gamma dist
    sorted_counts_bernie = sorted(num_retweets_bernie, reverse=True)
    shape, loc, scale = scistats.invgamma.fit(sorted_counts_bernie)
    rv = scistats.invgamma(shape, loc, scale)

    # Freedman–Diaconis rule for histogram bin selection
    iqr = scistats.iqr(sorted_counts_bernie)
    num_bins = int((2 * iqr) // np.cbrt(len(sorted_counts_bernie)))

    bernie_linspace = np.linspace(0, max(sorted_counts_bernie))
    axs[1, 0].hist(sorted_counts_bernie, bins=num_bins, density=True)
    axs[1, 0].plot(bernie_linspace, rv.pdf(bernie_linspace))
    axs[1, 0].set_title('Bernie Retweet Counts')

    # Create Yang plot

    num_retweets_yang = np.array([yang_tweets[i]['retweet_count']
                                  for i in range(len(yang_tweets))])

    # Estimate parameters of inverse gamma dist
    sorted_yang_counts = sorted(num_retweets_yang, reverse=True)
    shape, loc, scale = scistats.invgamma.fit(sorted_yang_counts)
    rv = scistats.invgamma(shape, loc, scale)

    # Freedman–Diaconis rule for histogram bin selection
    iqr = scistats.iqr(sorted_yang_counts)
    num_bins = int((2 * iqr) // np.cbrt(len(sorted_yang_counts)))

    yang_linspace = np.linspace(0, max(sorted_yang_counts))
    axs[1, 1].hist(sorted_yang_counts, bins=num_bins, density=True)
    axs[1, 1].plot(yang_linspace, rv.pdf(yang_linspace))
    axs[1, 1].set_title('Yang Retweet Counts')

    # Label the axes of all of the plots and save the plot to a file
    for ax in axs.flat:
        ax.set(xlabel='Number of Retweets', ylabel='Tweet Counts')

    plt.savefig(f'{image_dir}/retweet_counts_analysis.png')

    # Aggregate all of the keywords for each tweet of each candidate and
    # create a word cloud for each candidate out of their aggregated keywords
    # This is done by creating a single space-separated string of all of the
    # keywords across all of the tweets analyzed in the sentiment analysis. We
    # ignore the word amp, since it is just means ampersand and provides no
    # meaning.

    # Load the sentiment scores of all of the candidates

    warren_senti_scores = pickle.load(
        open(f'{directory}/senti_scores/warren_senti_scores.pkl', 'rb'))
    biden_senti_scores = pickle.load(
        open(f'{directory}/senti_scores/biden_senti_scores.pkl', 'rb'))
    bernie_senti_scores = pickle.load(
        open(f'{directory}/senti_scores/bernie_senti_scores.pkl', 'rb'))
    yang_senti_scores = pickle.load(
        open(f'{directory}/senti_scores/yang_senti_scores.pkl', 'rb'))

    # Warren's Word Cloud

    keyword_string = ' '.join([keyword['text'].lower()
                               for score in warren_senti_scores.values()
                               for keyword in score['keywords']
                               if keyword['text'].lower() != 'amp'])

    wordcloud = WordCloud(max_words=100, scale=2).generate(keyword_string)
    plt.figure(figsize=[15, 15])
    plt.axis("off")
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.savefig(f'{image_dir}/warren_word_cloud.png')

    # Biden's Word Cloud

    keyword_string = ' '.join([keyword['text'].lower()
                               for score in biden_senti_scores.values()
                               for keyword in score['keywords']
                               if keyword['text'].lower() != 'amp'])

    wordcloud = WordCloud(max_words=100, scale=2).generate(keyword_string)
    plt.figure(figsize=[15, 15])
    plt.axis("off")
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.savefig(f'{image_dir}/biden_word_cloud.png')

    # Bernie's Word Cloud

    keyword_string = ' '.join([keyword['text'].lower()
                               for score in bernie_senti_scores.values()
                               for keyword in score['keywords']
                               if keyword['text'].lower() != 'amp'])

    wordcloud = WordCloud(max_words=100, scale=2).generate(keyword_string)
    plt.figure(figsize=[15, 15])
    plt.axis("off")
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.savefig(f'{image_dir}/bernie_word_cloud.png')

    # Yang's Word Cloud

    keyword_string = ' '.join([keyword['text'].lower()
                               for score in yang_senti_scores.values()
                               for keyword in score['keywords']
                               if keyword['text'].lower() != 'amp'])

    wordcloud = WordCloud(max_words=100, scale=2).generate(keyword_string)
    plt.figure(figsize=[15, 15])
    plt.axis("off")
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.savefig(f'{image_dir}/yang_word_cloud.png')


@main.command('stats')
@click.argument('directory', type=click.Path(exists=True))
def stats(directory):
    """
    Read all data and print statistics. The directory argument in the case
    of this project is the data directory at the root of the project, since all
    of the tweets are stored there, so we can run "osna stats data" from the root
    of the repo to read all of the statistics.
    E.g., how many messages/users, time range, number of terms/tokens, etc.
    """
    print('reading from %s' % directory)
    # use glob to iterate all files matching desired pattern (e.g., .json files).
    # recursively search subdirectories.
    count_tweets = 0
    tweet_files = glob.glob(f'{directory}/tweets/old/*')
    for tweet_file in tweet_files:
        tweets = pickle.load(open(tweet_file, 'rb'))
        # All of the tweets in a file belong to a single candidate
        user = tweets[0]['user']['screen_name']
        count_tweets += len(tweets)
        print(f'Found a total of {len(tweets)} old tweets for user {user}')

    new_tweets_count = 0
    new_tweet_files = glob.glob(f'{directory}/tweets/new/*')
    for tweet_file in new_tweet_files:
        tweets = pickle.load(open(tweet_file, 'rb'))
        user = tweets[0]['user']['screen_name']
        new_tweets_count += len(tweets)
        print(f'Found a total of {len(tweets)} new tweets for user {user}')

    print(f'Found {count_tweets + new_tweets_count} tweets in total')


@main.command('train')
@click.argument('directory', type=click.Path(exists=True))
def train(directory):
    """
    Train a classifier on all of your labeled data and save it for later
    use in the web app. You should use the pickle library to read/write
    Python objects to files. You should also reference the `clf_path`
    variable, defined in __init__.py, to locate the file.
    """
    print('reading from %s' % directory)

    # 1- Load tweets and sentiment scores of all of the candidates.
    warren_tweets = pickle.load(
        open(f'{directory}/tweets/old/warren_tweets_old.pkl', 'rb'))
    biden_tweets = pickle.load(
        open(f'{directory}/tweets/old/biden_tweets_old.pkl', 'rb'))
    bernie_tweets = pickle.load(
        open(f'{directory}/tweets/old/bernie_tweets_old.pkl', 'rb'))
    yang_tweets = pickle.load(
        open(f'{directory}/tweets/old/yang_tweets_old.pkl', 'rb'))

    warren_senti_scores = pickle.load(
        open(f'{directory}/senti_scores/warren_senti_scores.pkl', 'rb'))
    biden_senti_scores = pickle.load(
        open(f'{directory}/senti_scores/biden_senti_scores.pkl', 'rb'))
    bernie_senti_scores = pickle.load(
        open(f'{directory}/senti_scores/bernie_senti_scores.pkl', 'rb'))
    yang_senti_scores = pickle.load(
        open(f'{directory}/senti_scores/yang_senti_scores.pkl', 'rb'))

    # 2- Find max character
    warren_max_char=0 #316
    for i in warren_tweets[0:]:
        warren_max_char = max(warren_max_char, warren_senti_scores[i['full_text']]['usage']['text_characters'])
    biden_max_char=0 #315
    for i in biden_tweets[0:]:
        biden_max_char = max(biden_max_char, biden_senti_scores[i['full_text']]['usage']['text_characters'])
    bernie_max_char=0 # 304
    for i in bernie_tweets:
        bernie_max_char = max(bernie_max_char, bernie_senti_scores[i['full_text']]['usage']['text_characters'])
    yang_max_char= 0 #329
    for i in yang_tweets:
        yang_max_char = max(yang_max_char, yang_senti_scores[i['full_text']]['usage']['text_characters'])

    # 3- Create corpus for each candidate
    # Warren:
    corpus = set()
    for tweet in warren_senti_scores:
        corpus.update({i['text'] for i in warren_senti_scores[tweet]['keywords']})
    warren_sorted_corpus = sorted(corpus)

    with open(f'{directory}/features/warren_corpus.pk', 'wb') as file:
        pickle.dump(warren_sorted_corpus, file)

    # Biden
    corpus = set()
    for tweet in biden_senti_scores:
        corpus.update({i['text'] for i in biden_senti_scores[tweet]['keywords']})
    biden_sorted_corpus = sorted(corpus)

    with open(f'{directory}/features/biden_corpus.pk', 'wb') as file:
        pickle.dump(biden_sorted_corpus, file)

    # Bernie
    corpus = set()
    for tweet in bernie_senti_scores:
        corpus.update({i['text'] for i in bernie_senti_scores[tweet]['keywords']})
    bernie_sorted_corpus = sorted(corpus)

    with open(f'{directory}/features/bernie_corpus.pk', 'wb') as file:
        pickle.dump(bernie_sorted_corpus, file)

    # Yang:
    corpus = set()
    for tweet in yang_senti_scores:
        corpus.update({i['text'] for i in yang_senti_scores[tweet]['keywords']})
    yang_sorted_corpus = sorted(corpus)

    with open(f'{directory}/features/yang_corpus.pk', 'wb') as file:
        pickle.dump(yang_sorted_corpus, file)

    # 4- Extract features and labels.
    # Each candidate has their own way of extracting features.

    # Warren:
    # Create a feature matrix
    warren_features = []
    warren_labels = []
    warren_feature_names = ['sadness', 'joy', 'fear', 'disgust', 'anger',
                            'sentiment', 'character'] + [i for i in warren_sorted_corpus]

    for i in warren_tweets:
        # Binary Labels
        if i['retweet_count'] <= 1083:
            warren_labels.append(-1)
        elif i['retweet_count'] >= 1614:
            warren_labels.append(1)
        else: # Discard ambigious tweets.
            continue

        # Feature
        tweet_feature = []
        for j,k in warren_senti_scores[i['full_text']]['emotion']['document']['emotion'].items():
            tweet_feature.append(k)
        tweet_feature.append(warren_senti_scores[i['full_text']]['sentiment']['document']['score'])
        warren_feature_names.append('sentiment')

        tweet_feature.append(warren_senti_scores[i['full_text']]['usage']['text_characters']/warren_max_char)
        warren_feature_names.append('character')

        # One-hot Encoded Features
        text_relevance = dict({sent['text']:sent['relevance'] for sent in warren_senti_scores[i['full_text']]['keywords']})
        tweet_onehot=[]
        for keys in warren_sorted_corpus:

            tweet_onehot.append(0 if keys not in text_relevance.keys() else text_relevance[keys])
        tweet_feature.extend(tweet_onehot)

        # Add all to features matrix
        warren_features.append(tweet_feature)

    with open(f'{directory}/features/warren_features.pk', 'wb') as file:
        pickle.dump([warren_features, warren_feature_names, warren_labels], file)


    # Biden:
    # Create a feature matrix
    biden_features = []
    biden_labels = []
    biden_feature_names = ['sadness', 'joy', 'fear', 'disgust', 'anger',
                            'sentiment', 'character'] + [i for i in biden_sorted_corpus]

    for i in biden_tweets:
        # Ambigious discarded Binary Labels
        if i['retweet_count'] <= 208:
            biden_labels.append(-1)
        elif i['retweet_count'] >= 302:
            biden_labels.append(1)
        else:
            continue

        # Feature
        tweet_feature = []
        for j,k in biden_senti_scores[i['full_text']]['emotion']['document']['emotion'].items():
            tweet_feature.append(k)
        tweet_feature.append(biden_senti_scores[i['full_text']]['sentiment']['document']['score'])
        biden_feature_names.append('sentiment')

        tweet_feature.append(biden_senti_scores[i['full_text']]['usage']['text_characters']/biden_max_char)
        biden_feature_names.append('character')

        # One-hot Encoded Features
        text_relevance = dict({sent['text']:sent['relevance'] for sent in biden_senti_scores[i['full_text']]['keywords']})
        tweet_onehot=[]
        for keys in biden_sorted_corpus:

            tweet_onehot.append(0 if keys not in text_relevance.keys() else text_relevance[keys])
        tweet_feature.extend(tweet_onehot)

        # Add all to features matrix
        biden_features.append(tweet_feature)

    with open(f'{directory}/features/biden_features.pk', 'wb') as file:
        pickle.dump([biden_features, biden_feature_names, biden_labels], file)

    # Bernie
    # Create a feature matrix
    bernie_features = []
    bernie_labels = []
    bernie_feature_names = ['sadness', 'joy', 'fear', 'disgust', 'anger',
                            'sentiment', 'character'] + [i for i in bernie_sorted_corpus]

    for i in bernie_tweets:
        # Binary Labels
        if i['retweet_count'] <= 1080:
            bernie_labels.append(-1)
        elif i['retweet_count'] >= 1612:
            bernie_labels.append(1)
        else: # Ambigious labels discarded.
            continue

        # Feature
        tweet_feature = []
        for j,k in bernie_senti_scores[i['full_text']]['emotion']['document']['emotion'].items():
            tweet_feature.append(k)
        tweet_feature.append(bernie_senti_scores[i['full_text']]['sentiment']['document']['score'])
        bernie_feature_names.append('sentiment')

        tweet_feature.append(bernie_senti_scores[i['full_text']]['usage']['text_characters']/bernie_max_char)
        bernie_feature_names.append('character')

        # One-hot Encoded Features
        text_relevance = dict({sent['text']:sent['relevance'] for sent in bernie_senti_scores[i['full_text']]['keywords']})
        tweet_onehot=[]
        for keys in bernie_sorted_corpus:

            tweet_onehot.append(0 if keys not in text_relevance.keys() else text_relevance[keys])
        tweet_feature.extend(tweet_onehot)

        # Add all to features matrix
        bernie_features.append(tweet_feature)

    with open(f'{directory}/features/bernie_features.pk', 'wb') as file:
        pickle.dump([bernie_features, bernie_feature_names, bernie_labels], file)

    # Yang:
    # Create a feature matrix
    yang_features = []
    yang_labels = []
    yang_feature_names = ['sadness', 'joy', 'fear', 'disgust', 'anger',
                            'sentiment', 'character'] + [i for i in yang_sorted_corpus]

    for i in yang_tweets:
        # Ambigious discarded Binary Labels
        if i['retweet_count'] <= 335: #880:
            yang_labels.append(-1)
        elif i['retweet_count'] >= 524: #1612:
            yang_labels.append(1)
        else:
            continue

        # Feature
        tweet_feature = []
        for j,k in yang_senti_scores[i['full_text']]['emotion']['document']['emotion'].items():
            tweet_feature.append(k)
        tweet_feature.append(yang_senti_scores[i['full_text']]['sentiment']['document']['score'])
        yang_feature_names.append('sentiment')

        tweet_feature.append(yang_senti_scores[i['full_text']]['usage']['text_characters']/yang_max_char)
        yang_feature_names.append('character')

        # One-hot Encoded Features
        text_relevance = dict({sent['text']:sent['relevance'] for sent in yang_senti_scores[i['full_text']]['keywords']})
        tweet_onehot=[]
        for keys in yang_sorted_corpus:

            tweet_onehot.append(0 if keys not in text_relevance.keys() else text_relevance[keys])
        tweet_feature.extend(tweet_onehot)

        # Add all to features matrix
        yang_features.append(tweet_feature)

    with open(f'{directory}/features/biden_features.pk', 'wb') as file:
        pickle.dump([yang_features, yang_feature_names, yang_labels], file)




    # 5 - Split the data to train and test
    X_train_warren, X_test_warren, y_train_warren, y_test_warren = train_test_split(warren_features, warren_labels, test_size=1/3, random_state=42)
    X_train_biden, X_test_biden, y_train_biden, y_test_biden = train_test_split(biden_features, biden_labels, test_size=1/3, random_state=42)
    X_train_bernie, X_test_bernie, y_train_bernie, y_test_bernie = train_test_split(bernie_features, bernie_labels, test_size=1/3, random_state=42)
    X_train_yang, X_test_yang, y_train_yang, y_test_yang = train_test_split(yang_features, yang_labels, test_size=1/3, random_state=42)

    # 6 - Create classifier for each candidate
    lr_warren = LogisticRegression(C=2.0)
    lr_warren.fit(X_train_warren, y_train_warren)

    lr_biden = LogisticRegression(C=2.0)
    lr_biden.fit(X_train_biden, y_train_biden)

    lr_bernie = LogisticRegression(C=2.0)
    lr_bernie.fit(X_train_bernie, y_train_bernie)

    lr_yang = LogisticRegression(C=2.0)
    lr_yang.fit(X_train_yang, y_train_yang)

    # 7 - Dump evaluation results.
    warren_train_acc = lr_warren.score(X_train_warren, y_train_warren)
    warren_test_acc = lr_warren.score(X_test_warren, y_test_warren)
    warren_train_f1 = f1_score(lr_warren.predict(X_test_warren), y_test_warren)
    warren_test_f1 = f1_score(lr_warren.predict(X_train_warren), y_train_warren)
    with open(f'{directory}/evaluate/warren_evaluate.pk', 'wb') as file:
        pickle.dump([warren_train_acc, warren_test_acc, warren_train_f1, warren_test_f1], file)

    biden_train_acc = lr_biden.score(X_train_biden, y_train_biden)
    biden_test_acc = lr_biden.score(X_test_biden, y_test_biden)
    biden_train_f1 = f1_score(lr_biden.predict(X_test_biden), y_test_biden)
    biden_test_f1 = f1_score(lr_biden.predict(X_train_biden), y_train_biden)
    with open(f'{directory}/evaluate/biden_evaluate.pk', 'wb') as file:
        pickle.dump([biden_train_acc, biden_test_acc, biden_train_f1, biden_test_f1], file)

    bernie_train_acc = lr_bernie.score(X_train_bernie, y_train_bernie)
    bernie_test_acc = lr_bernie.score(X_test_bernie, y_test_bernie)
    bernie_train_f1 = f1_score(lr_bernie.predict(X_test_bernie), y_test_bernie)
    bernie_test_f1 = f1_score(lr_bernie.predict(X_train_bernie), y_train_bernie)
    with open(f'{directory}/evaluate/bernie_evaluate.pk', 'wb') as file:
        pickle.dump([bernie_train_acc, bernie_test_acc, bernie_train_f1, bernie_test_f1], file)

    yang_train_acc = lr_yang.score(X_train_yang, y_train_yang)
    yang_test_acc = lr_yang.score(X_test_yang, y_test_yang)
    yang_train_f1 = f1_score(lr_yang.predict(X_test_yang), y_test_yang)
    yang_test_f1 = f1_score(lr_yang.predict(X_train_yang), y_train_yang)
    with open(f'{directory}/evaluate/yang_evaluate.pk', 'wb') as file:
        pickle.dump([yang_train_acc, yang_test_acc, yang_train_f1, yang_test_f1], file)


    # 8 - Dump the classifiers
    with open(f'{directory}/clf/warren_lr.pk', 'wb') as file:
        pickle.dump(lr_warren, file)
    with open(f'{directory}/clf/biden_lr.pk', 'wb') as file:
        pickle.dump(lr_biden, file)
    with open(f'{directory}/clf/bernie_lr.pk', 'wb') as file:
        pickle.dump(lr_bernie, file)
    with open(f'{directory}/clf/yang_lr.pk', 'wb') as file:
        pickle.dump(lr_yang, file)




@main.command('web')
@click.option('-t', '--twitter-credentials', required=False, type=click.Path(exists=True), show_default=True, default=credentials_path, help='a json file of twitter tokens')
@click.option('-p', '--port', required=False, default=9999, show_default=True, help='port of web server')
def web(twitter_credentials, port):
    """
    Launch a web app for your project demo.
    """
    from .app import app
    app.run(host='0.0.0.0', debug=True, port=port)


####################
# HELPER FUNCTIONS #
####################

# Convenience functions for collecting data from twitter


def get_twitter():
    """ Construct an instance of TwitterAPI using the tokens in the file at credentials_path
    Returns:
      An instance of TwitterAPI.
    """
    twitter_creds = json.load(open(credentials_path))['Twitter']
    consumer_key = twitter_creds['consumer_key']
    consumer_secret = twitter_creds['consumer_secret']
    access_token = twitter_creds['access_token']
    access_token_secret = twitter_creds['access_token_secret']
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)


def get_ibm_nlu():
    """ Constructs an instance of NaturalLanguageUnderstandingV1 using the tokens in the file
    at credentials_path
    Returns:
        An instance of NaturalLanguageUnderstandingV1
    """
    ibm_creds = json.load(open(credentials_path))['IBM-Cloud']
    authenticator = IAMAuthenticator(ibm_creds['api_key'])
    nlu = NaturalLanguageUnderstandingV1(
        version='2019-07-12',
        authenticator=authenticator
    )
    nlu.set_service_url(ibm_creds['service_url'])
    return nlu


def robust_request(twitter, resource, params, max_tries=5):
    """ If a Twitter request fails, sleep for 15 minutes.
    Do this at most max_tries times before quitting.
    Args:
      twitter .... A TwitterAPI object.
      resource ... A resource string to request; e.g., "friends/ids"
      params ..... A parameter dict for the request, e.g., to specify
                   parameters like screen_name or count.
      max_tries .. The maximum number of tries to attempt.
    Returns:
      A TwitterResponse object, or None if failed.
    """
    for i in range(max_tries):
        request = twitter.request(resource, params)
        if request.status_code == 200:
            return request
        else:
            print('Got error %s \nsleeping for 15 minutes.' % request.text)
            sys.stderr.flush()
            time.sleep(61 * 15)


def get_timeline_for_candidate(twitter, limit, candidate_name):
    tweets = []
    initial_response = robust_request(twitter, 'statuses/user_timeline',
                                      {'screen_name': candidate_name,
                                       'count': limit,
                                       'include_rts': False,
                                       'tweet_mode': 'extended'})
    tweets.extend([tweet for tweet in initial_response])
    if len(tweets) >= limit:
        return tweets

    # We subtract 1 in order to not get redundant tweets on the next request
    min_id = min([tweet['id'] for tweet in initial_response]) - 1
    while True:
        response = robust_request(twitter, 'statuses/user_timeline',
                                  {'screen_name': candidate_name,
                                   'max_id': min_id,
                                   'include_rts': False,
                                   'tweet_mode': 'extended'})
        tweets.extend([tweet for tweet in response])
        print(f'Number of tweets found so far: {len(tweets)}')
        if len(tweets) >= limit:
            return tweets[:limit]
        try:
            min_id = min([tweet['id'] for tweet in response]) - 1
        except:
            print('Tweet limit from API reached, returning all of the tweets retrieved')
            return tweets


# Helper function for sentiment analysis

def sentiment_analysis(tweets_list, nlu):
    """
    Args:
      tweets_list .... The plain text list of tweets for a candidate
      nlu ... The natural language understanding tool from IBM Watson
    Returns:
      A python dictionary mapping the tweet text to all of the sentiment scores
    """
    senti_scores = {}
    for i, tweet in enumerate(tweets_list):
        try:
            tweet_scores = nlu.analyze(text=tweets_list[i], features=Features(
                emotion=EmotionOptions(),
                sentiment=SentimentOptions(),
                keywords=KeywordsOptions()), language='en').get_result()
            senti_scores[tweet] = tweet_scores
        except:
            senti_scores[tweet] = 'N/A, could not parse'

    return senti_scores


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
