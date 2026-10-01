"""Emotion Tagger Data Analysis

Pre-requisites + Loading in Data
"""

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import sys
sys.path.append('/content/drive/MyDrive/Colab Notebooks')

import time
import read_write_vecs

loading_vecs = True
if loading_vecs:
    base_path = '/content/drive/MyDrive/Colab Notebooks/'

    binary_path = base_path + 'wvecs.npy'
    words_path = base_path + 'vocab.txt'
    categories_path = base_path + 'emotion_categories.txt'
    practice_tweets = base_path + 'random_tweet_selection_train_1-2.txt'
    accuracy_information = base_path + 'accuracy_score_information.txt'
    big_tweetin = base_path + 'random_tweet_selection_train2.txt'
    big_answerin = base_path + 'random_tweet_codes_train_1.txt'

    start = time.time()
    wvecs = read_write_vecs.read_binary_n(fn=binary_path)
    words = read_write_vecs.read_words(words_path)
    categories = read_write_vecs.read_categories(categories_path)
    end = time.time()
    practiceTweets = read_write_vecs.read_words(practice_tweets)
    accuracy_info = read_write_vecs.read_words(accuracy_information)
    big_tweets = read_write_vecs.read_words(big_tweetin)
    big_answers = read_write_vecs.read_words(big_answerin)

"""Embedding"""

import ast

emotionsGold = []

for s in big_answers:
    s = s.strip().lstrip('\ufeff')
    emotion2, score = ast.literal_eval(s)
    emotionsGold.append(emotion2)
print(emotionsGold)

bigGoldData = []

for i in range(0, len(big_tweets)):
    tweet_encoding2 = big_tweets[i]
    label2          = emotionsGold[i]
    bigGoldData.append((tweet_encoding2, label2))

!pip install transformers pandas numpy

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-uncased")
hf_token = "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"

emotionNumbers = {'sadness': 1, 'fear': 2, 'anger': 3, 'joy': 4}

numberedTweets = [(text, emotionNumbers.get(emotion, 0)) for text, emotion in bigGoldData]

trainingTweets = [text for text, label in bigGoldData]

justEncodings = tokenizer(
    trainingTweets,
    return_tensors='pt',
    padding=True,
    truncation=True
)

input_ids = justEncodings["input_ids"]

labeling = [(emotionNumbers.get(emotion, 0)) for text, emotion in bigGoldData]

encodedTweetEmotion = []

for i in range(0, len(trainingTweets)):
    tweet_encoding = input_ids[i]
    label          = labeling[i]
    encodedTweetEmotion.append((tweet_encoding, label))

x = [t[0] for t in encodedTweetEmotion]
y = [t[1] for t in encodedTweetEmotion]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=47)

Frf = RandomForestClassifier(bootstrap = True, max_depth = 10, max_features = 0.5, min_samples_leaf = 2, min_samples_split = 4, n_estimators = 200)
Frf.fit(x_train, y_train)
train_predictions = Frf.predict(x_train)
test_predictions = Frf.predict(x_test)
train_acc = accuracy_score(y_train, train_predictions)
test_acc = accuracy_score(y_test, test_predictions)
print('train acc', train_acc)
print('test acc', test_acc)

"""Kagle API Token

export KAGGLE_API_TOKEN=***

kaggle competitions list

Password - ***

Data Analysis
"""

import kagglehub

path = kagglehub.dataset_download("rohanroy1/2024-us-presidential-elections-twitter-data")

print("Path to dataset files:", path)

print("Path to dataset files:", path)
print("Files in that folder:")
print(os.listdir(path))

csv_path = os.path.join(path, "preprocessedtranslated_tweets_us24.csv")
df = pd.read_csv(csv_path)
df.head()

df.columns

tweets24 = df["Text"].dropna().astype(str).tolist()

tweets24[200]

tweets24Encodings = tokenizer(
    tweets24,
    return_tensors='pt',
    padding=True,
    truncation=True,
    max_length = 64
)

X_new = tweets24Encodings["input_ids"]
y_pred2 = Frf.predict(X_new)

y_pred2
len(y_pred2)

y_pred2

"""Key to remember - 'sadness: 1', 'fear: 2', 'anger: 3', 'joy: 4'"""

from collections import Counter

counts = Counter(y_pred2)
print(counts)

for i in range(1, 5):
    print(i, counts[i])

"""so that's
- 13,661 fear
- 6,553 anger
- 3162 joy
- 2976 sadness
"""

import matplotlib.pyplot as plt

emotions24 = ['sadness: 1', 'fear: 2', 'anger: 3', 'joy: 4']
entries24 = [2976, 13661, 6554, 3162]

plt.bar(emotions24, entries24, color = 'lightpink')
plt.title('Emotions 2024 Election')
plt.xlabel('Emotions')
plt.ylabel('Entries')
plt.show()

sadness24 = 2976/26353
fear24 = 13661/26353
anger24 = 6554/26353
joy24 = 3162/26353

"""2020 Data"""

import kagglehub

# Download latest version
path3 = kagglehub.dataset_download("sripaadsrinivasan/tweets-about-the-upcoming-us-electionaugtooct")

print("Path to dataset files:", path)

print("Path to dataset files:", path3)
print("Files in that folder:")
print(os.listdir(path3))

csv_path3 = os.path.join(path3, 'election2020.csv')
df2 = pd.read_csv(csv_path3, engine="python", on_bad_lines="skip" )
df2.head()

df2.columns

df2['tweet']

tweets20 = df2['tweet'].dropna().astype(str).tolist()

tweets20Encodings = tokenizer(
    tweets20,
    return_tensors='pt',
    padding=True,
    truncation=True,
    max_length = 64
)

X_new20 = tweets20Encodings["input_ids"]
y_pred20 = Frf.predict(X_new20)

from collections import Counter

counts20 = Counter(y_pred20)
print(counts20)

"""Emotions as follows
- sadness 11897
- fear 343942
- anger 28107
- joy 13015
"""

import matplotlib.pyplot as plt

emotions20 = ['sadness: 1', 'fear: 2', 'anger: 3', 'joy: 4']
entries20 = [11897, 343942, 28107, 13015]

plt.bar(emotions20, entries20, color = 'lightpink')
plt.title('Emotions 2020 Election')
plt.xlabel('Emotions')
plt.ylabel('Entries')
plt.show()

sadness20 = 11897/396961
fear20 = 343942/396961
anger20 = 28107/396961
joy20 = 13015/396961

"""Side by side comparison"""

import numpy as np
import matplotlib.pyplot as plt

cats = ['sadness', 'fear', 'anger', 'joy']
v1, v2 = [sadness20, fear20, anger20, joy20], [sadness24, fear24, anger24, joy24]
w, x = 0.4, np.arange(len(cats))

plt.bar(x - w/2, v1, w, label='2020 Election', color = 'lightpink')
plt.bar(x + w/2, v2, w, label='2024 Election', color='lightblue')

plt.xticks(x, cats)
plt.ylabel('Percentage of total emotions for the year')
plt.title('Emotions by Percentage 2020 vs 2024 Election')
plt.legend()
plt.show()

"""The difference is statistically signficant when looking at a relation based t-test"""

from scipy.stats import ttest_rel

t_stat, p_value = ttest_rel(entries20, entries24)
print("t =", t_stat)
print("p =", p_value)

"""Findings
- Fear dropped by a total of 34.8%
- There is a strong statisttically sigificant difference in the emotions between the 2020 election and the 2024 election
- We saw sadness increase by 8.3% from 2020 - 2024
- We also saw joy increase by 8.7% from 2020 - 2024
"""
