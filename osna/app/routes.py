import json
import os
import pickle
import sys

from flask import render_template, flash, redirect, session, request
from sklearn.linear_model import LogisticRegression
from . import app
from .forms import MyForm
from .predictor_api import sentiment_analysis, extract_features, return_corpus_and_model_name, candidate_predict, get_ibm_nlu
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, EmotionOptions, SentimentOptions, KeywordsOptions
from .. import credentials_path

# Paths
FeatureDataPath = os.path.join(app.root_path, 'FeatureData')
PredicationsPath = os.path.join(app.root_path, 'Predictions')


@app.after_request
def add_header(response):
    response.cache_control.no_store = True
    return response


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    session.pop('_flashes', None)
    form = MyForm()
    result = None
    nlu = get_ibm_nlu()

    if form.validate_on_submit():
        input_tweet = form.input_field.data
        candidate = request.form['options']
        candidate_corpus_filename, candidate_model = return_corpus_and_model_name(
            candidate)
        print(candidate_corpus_filename, candidate_model)

        with open(FeatureDataPath + '/'+candidate_corpus_filename, 'rb') as corpusfile:
            corpus = pickle.load(corpusfile)

        with open(PredicationsPath + '/'+candidate_model, 'rb') as modelfile:
            model = pickle.load(modelfile)

        senti_result = sentiment_analysis(input_tweet, nlu)
        tweet_features, tweet_feature_names = extract_features(
            input_tweet, senti_result, corpus)
        candidate_prediction = candidate_predict(tweet_features, model)
        print(candidate_prediction)
        # take the first element of the array
        candidate_prediction = str(candidate_prediction[0])
        print(candidate_prediction)
        return render_template('result.html',
                               title='',
                               form=form,
                               response=senti_result,
                               input_tweet=input_tweet,
                               candidate=candidate,
                               reach_prediction=candidate_prediction)
    return render_template('myform.html', title='', form=form)
