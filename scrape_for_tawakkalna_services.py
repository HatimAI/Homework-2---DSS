# -*- coding: utf-8 -*-
"""Scrape for Tawakkalna Services.ipynb
"""

!pip install google-play-scraper
!pip install -q transformers
!pip install -q plotly-express

import pandas as pd
import numpy as np
from google_play_scraper import app, Sort, reviews_all
import plotly.express as px
import tensorflow as tf

gp_project = reviews_all('sa.gov.nic.twkhayat', sleep_milliseconds=0, lang='en', country='SA', sort=Sort.NEWEST)
df_reviews = pd.DataFrame(np.array(gp_project), columns=['review'])
df_reviews = df_reviews.join(pd.DataFrame(df_reviews.pop('review').tolist()))
df_reviews.head()

df = pd.json_normalize(gp_project)

from transformers import pipeline
sentiment_analysis = pipeline("sentiment-analysis",model="siebert/sentiment-roberta-large-english")

df['content'] = df['content'].astype('str')

df['result'] = df['content'].apply(lambda x: sentiment_analysis(x))

df['Sentiment'] = df['result'].apply(lambda x: (x[0]['label']))
df['Sentiment Score'] = df['result'].apply(lambda x: (x[0]['score']))

import pandas as pd
import re

OnlyEnglish_regex = re.compile(r'[\u0600-\u06FF\u0900-\u097F\u0980-\u09FF\u0C00-\u0C7F\u0B00-\u0B7F]')

OnlyEnglish_rows = df[df['content'].str.contains(OnlyEnglish_regex)]

df = df.drop(OnlyEnglish_rows.index)

df = df.loc[:, ['reviewId', 'userName', 'content', 'score', 'thumbsUpCount', 'at', 'appVersion', 'Sentiment', 'Sentiment Score']]

df.to_csv('GooglePlayReviews.csv', index=False)

df.head()

df.head()

