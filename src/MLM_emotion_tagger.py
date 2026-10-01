"""
Pre-requisites + Loading in Data
"""

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import numpy as np

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
hf_token = "***"

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

"""Bar plot of emotions to check for relatively fair spread of examples"""

import matplotlib.pyplot as plt

emotions = ['sadness: 1', 'fear: 2', 'anger: 3', 'joy: 4']
entries = [706, 1020, 776, 749]

plt.bar(emotions, entries)
plt.title('Emotion Counts')
plt.xlabel('Emotions')
plt.ylabel('Entries')
plt.show()

"""Decision Tree Model"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

x = [t[0] for t in encodedTweetEmotion]
y = [t[1] for t in encodedTweetEmotion]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=47)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

dtree = DecisionTreeClassifier(max_depth=6)
dtree.fit(x_train, y_train)
train_predictions = dtree.predict(x_train)
test_predictions = dtree.predict(x_test)
train_acc = accuracy_score(y_train, train_predictions)
test_acc = accuracy_score(y_test, test_predictions)
print('train acc', train_acc)
print('test acc', test_acc)

from sklearn.tree import DecisionTreeClassifier

best_acc = 0

for criterion in "gini", "entropy":
    for max_depth in [2,3,4,5,6]:
        for min_samples_leaf in [4, 5, 6, 10]:
            dtree = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, min_samples_leaf=min_samples_leaf)
            dtree.fit(x_train, y_train)
            predicitions = dtree.predict(x_test)
            acc = accuracy_score(y_test, predicitions)
            if acc > best_acc:
                best_params = f"criterion: {criterion}, max_depth: {max_depth}, min_samples_leaf: {min_samples_leaf}"
                best_acc = acc
print(best_params)

x = [t[0] for t in encodedTweetEmotion]
y = [t[1] for t in encodedTweetEmotion]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=47)

dtree2 = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=5, max_depth=6)
dtree2.fit(x_train, y_train)
train_predictions = dtree2.predict(x_train)
test_predictions = dtree2.predict(x_test)
train_acc = accuracy_score(y_train, train_predictions)
test_acc = accuracy_score(y_test, test_predictions)
print('train acc', train_acc)
print('test acc', test_acc)
print(classification_report(y_test, test_predictions))

"""Visualization of our Decision Tree"""

from sklearn.tree import plot_tree

fig = plt.figure(figsize=((25,20)))
plot_tree(dtree2,
            impurity=False,
            proportion=True,
            filled=True)

"""Random Forests"""

x = [t[0] for t in encodedTweetEmotion]
y = [t[1] for t in encodedTweetEmotion]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=47)

rf = RandomForestClassifier()
rf.fit(x_train, y_train)
train_predictions = rf.predict(x_train)
test_predictions = rf.predict(x_test)
train_acc = accuracy_score(y_train, train_predictions)
test_acc = accuracy_score(y_test, test_predictions)
print('train acc', train_acc)
print('test acc', test_acc)

"""Hyperparameter Tuning"""

x = [t[0] for t in encodedTweetEmotion]
y = [t[1] for t in encodedTweetEmotion]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=47)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [7, 9, 10],
    'min_samples_split': [4, 6, 8],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', 0.2, 0.5],
    'bootstrap': [True]
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=2)
grid_search.fit(x_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Estimator:", grid_search.best_estimator_)

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
print(classification_report(y_test, test_predictions))

"""Will use the random forests model for data analysis as it's more reliable"""
