# predictor_api.py

import json
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, EmotionOptions, SentimentOptions, KeywordsOptions
from .. import credentials_path


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


def sentiment_analysis(input_text, nlu):
    """
    Args:
    input_text .... The plain text of a tweet for a candidate
    nlu ... The natural language understanding tool from IBM Watson
    Returns:
    A python dictionary mapping the tweet text to all of the sentiment scores
    """
    tweet_score = nlu.analyze(text=input_text, features=Features(
        emotion=EmotionOptions(),
        sentiment=SentimentOptions(),
        keywords=KeywordsOptions()),
        language='en').get_result()
    return tweet_score


def sentiment_analysis_list(tweets_list, nlu):
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


def extract_features(tweet, senti_scores, candidate_corpus):
    """Extracts individual tweets' features.

        Args:
        - tweet: The entry to the API, String. 
        - senti_scores: The result of NLU
        Returns:
        - tweet_features: Values of features
        - tweet_feature_names: Corresponding names"""

    tweet_feature_names = ['sadness', 'joy', 'fear', 'disgust', 'anger',
                           'sentiment',
                           'character'] + [i for i in candidate_corpus]

    # Feature
    tweet_features = []
    for j, k in senti_scores['emotion']['document']['emotion'].items():  # Emotion
        tweet_features.append(k)
    tweet_features.append(
        senti_scores['sentiment']['document']['score'])  # Sentiment

    tweet_features.append(
        senti_scores['usage']['text_characters']/316)  # Max character

    # One-hot Encoded Features
    text_relevance = dict({sent['text']: sent['relevance']
                           for sent in senti_scores['keywords']})
    tweet_onehot = []
    for keys in candidate_corpus:
        tweet_onehot.append(
            0 if keys not in text_relevance.keys() else text_relevance[keys])
    tweet_features.extend(tweet_onehot)

    return tweet_features, tweet_feature_names
# Now you have yourself a tweet with its features.


def return_corpus_and_model_name(candidate):
    """Given full name of the candidate, returns corpus name and model_name

        Args:
        - candidate: Candidate name as per our html rendering, String. 
        Returns:
        - model_name: name of corpus file to use, string
        - corpus_name: name of corpus file to use, string"""
    if candidate == 'Bernie Sanders':
        corpus_name = 'bernie_corpus.pk'
        model_name = 'bernie_LR.pk'
    elif candidate == 'Andrew Yang':
        corpus_name = 'yang_corpus.pk'
        model_name = 'yang_LR.pk'
    elif candidate == 'Elizabeth Warren':
        corpus_name = 'warren_corpus.pk'
        model_name = 'warren_LR.pk'
    elif candidate == 'Joe Biden':
        corpus_name = 'biden_corpus.pk'
        model_name = 'biden_LR.pk'
    else:
        pass
    return corpus_name, model_name


def candidate_predict(tweet_features, candidate_clf):
    return candidate_clf.predict([tweet_features])
