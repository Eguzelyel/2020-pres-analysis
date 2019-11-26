## Overview

Different presidential candidates have different messaging; some present an optimistic vision for the future while some are more pessimistic and negative. We seek to analyze the tweets of certain democratic presidential candidates, classify them as positive and negative, then analyze the "reach" of the tweet(how many likes, RTs, replies, etc). We then wish to create a model, which when given a candidate and their tweet, it will estimate the "reach" of that tweet.



This is an interesting problem as during the 2016 elections, many believed Trump was running a somewhat pessimistic campaign but his ideals/vision resonated with many Americans and he ultimately became President. We wish to see which sort of ideas and statements resonate more with the candidate's followers; it gives some insight into the type of people that follow the candidate. In general, we want to see if Americans are more receptive to optimistic or pessimistic ideas.

## Data

• We will gather the tweets of various 2020 democratic presidential candidates (Warren, Bernie, Biden, Yang)

• We will use twitter API to pull this data

• We will search for Twitter data for sentiment analysis and possible models.



Problems we anticipate: 

• Not having enough tweets that are explicitly optimistic or pessimistic, so we wouldn't be able to classify them into either category

• By doing a binary classification, it might dilute the results. For example, a tweet that is minorly positive might not have as much reach as one that is extremely positive, so the "coefficient" wouldn't be as strong.

• We have to account for other variables that affect reach

• Tweets might not be the candidates own tweets but Retweets, or replies to other tweets. For example, if Trump tweets something negative and Bernie replies "Shame on you" it would be classified as negative but it's not really Bernie's own idea. Look more into twitter API.



## Method

• We will use the Natural Language ToolKit (NLTK) python library for Sentiment analysis and will not modify the code

• For the ML part, we will use numpy and pandas to create a model. We will try several different machine learning models (random forests, SVM, logistic regression, etc.) with different parameters and see which one yields the highest cross-validated score on training tweets before using it on the test dataset.

• Compare candidates separately to see how their reaches are affected. 

## Related Work

Analyzing Twitter Sentiment of the 2016 Presidential Candidates

https://web.stanford.edu/~jesszhao/files/twitterSentiment.pdf



Using sentiment analysis to define twitter political users’ classes and their homophily during the 2016 American presidential election

https://jisajournal.springeropen.com/articles/10.1186/s13174-018-0089-0



Sentiment Analysis of Tweets to Gain Insights into the 2016 US Election

https://cusj.columbia.edu/wp-content/uploads/sites/15/2017/06/Sentiment-Analysis-of-Tweets-to-Gain-Insights-into-the-2016-US-Election.pdf



Sentiment analysis of tweets for the 2016 US presidential election

https://ieeexplore.ieee.org/document/8284176



Location-Based Twitter Sentiment Analysis for Predicting the U.S. 2016 Presidential Election

https://pdfs.semanticscholar.org/dcf8/c45ef59cc0f30ce8bffbb6937102348de956.pdf



Twitter as a Corpus for Sentiment Analysis and Opinion Mining

https://lexitron.nectec.or.th/public/LREC-2010_Malta/pdf/385_Paper.pdf



Analyzing and Predicting Viral Tweets

http://delivery.acm.org/10.1145/2490000/2488017/p657-jenders.pdf?ip=104.194.125.154&id=2488017&acc=ACTIVE%20SERVICE&key=50864D773CC43BF0%2E50864D773CC43BF0%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&__acm__=1570052759_8dc36f681ecf39d6931b0e581aab9904 



## Evaluation

• To evaluate the results, we can calculate the F1 score performance metric, which is the harmonic mean of precision and recall of the test labels we predict for the reach of new tweets compared to their actual reach.

• Key plots: Barchart for each candidate showing average reach of each tweet category(positive vs negative)

• Predicted reach plot

• Correlation between negativity/positivity and popularity of candidate in polls across the nation

• We will use different models on classifying candidate tweets on optimism/pessimism and compare them using Cohen's Kappa statistic. 



