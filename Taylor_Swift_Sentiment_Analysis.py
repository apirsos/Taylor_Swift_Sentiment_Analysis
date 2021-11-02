#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 13:46:22 2021

@author: apirsos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from pathlib import Path
from PIL import Image

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import FreqDist

import pandas_bokeh
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from textblob import TextBlob
import seaborn as sns

pd.set_option('plotting.backend', 'pandas_bokeh')
pd.set_option('display.max_colwidth', 30)
pandas_bokeh.output_notebook()

import os
os.getcwd()

lyrics_df_2 = pd.read_csv('/Users/apirsos/Downloads/taylor_swift_lyrics.csv', encoding='Latin1')
lyrics_df_2.head()



#Data Preparation
#Make entire text lowercase
lyrics_df_2['lyric'] = [r.lower() for r in lyrics_df_2['lyric']]
#Remove unwanted characters
lyrics_df_2['lyric'] = lyrics_df_2['lyric'].str.replace("[^a-zA-Z#]"," ")
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
MyStopWords = ['like', 'ooh', 'im', 'youre', 'shake', 'oh', 'na', 'ill', 'ey', 'ive', 'id', 'hes', 'la', 'youll', 'youve', 'ha', 'uh', 'mm', 'theyll', 'ta', 'youd', 'ah', 'oohoohoohoohooh', 'theyre', 'ahaah', 'em', 'haah']
stop_words.extend(MyStopWords)

#Remove stop words
def remove_stopwords(rev):
    rev_new = " ".join([i for i in rev if i not in stop_words])
    return rev_new

#remove short words length < 2
lyrics_df_2['lyric'] = lyrics_df_2['lyric'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))

#remove stop words from text
lyrics_df_2['lyric'] = [remove_stopwords(r.split()) for r in lyrics_df_2['lyric']]

#drop empty lines
lyrics_df_2 = lyrics_df_2[lyrics_df_2['lyric']!=""]



lyrics_df_2['polarity'] = lyrics_df_2.apply(lambda x: 
        TextBlob(x['lyric']).sentiment.polarity, axis = 1)
lyrics_df_2.head()

#sentiment by line
sid = SentimentIntensityAnalyzer()
lyrics_df_2["sentiments"] = lyrics_df_2["lyric"].apply(lambda x:sid.polarity_scores(x))
lyrics_df_2 = pd.concat([lyrics_df_2.drop(['sentiments'], axis =1), 
lyrics_df_2['sentiments'].apply(pd.Series)], axis = 1)
lyrics_df_2.head()

lyrics_df_2.to_csv('shake_sentiment.csv')

#aggregate the lyrics by album
album_sent = lyrics_df_2.groupby("album").agg({'lyric':lambda x: " ".join(x), 'neg': 'mean', 'neu': 'mean', 'pos': 'mean', 'compound': 'mean'}).reset_index()

#filter by 1989 - album
nine = lyrics_df_2[lyrics_df_2.album=="1989"]
nine.to_csv('nine_sentiment.csv')
nine_song_sent = nine.groupby("track_title").agg({'lyric':lambda x: " ".join(x), 'neg': 'mean', 'neu': 'mean', 'pos': 'mean', 'compound': 'mean'}).reset_index()

    
def freq_words (x, terms = 20):
    all_words = ''.join([text for text in x] )
    all_words = all_words.split()
    fdist= FreqDist(all_words)
    words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})
    d = words_df.nlargest(columns='count', n = terms)
    plt.figure(figsize = (20,5))
    ax = sns.barplot(data = d, x = 'word', y = 'count')
    ax.set(ylabel = 'Count', xlabel = 'Word')
    plt.show()

freq_words(lyrics_df_2['lyric'])

freq_words(lyrics_df_2, 20)