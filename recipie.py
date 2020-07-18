import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

data = pd.read_csv('sample_submission.csv')
train_data = pd.read_json('train.json')
test_data = pd.read_json('test.json')

train_data.info()
train_data.isnull().sum()
train_data.shape
train_data.head()
train_data.tail()

# test data
test_data.info()
test_data.isnull().sum()
test_data.shape
test_data.head()
test_data.tail()

print("categories variables = {}".format(len(train_data.cuisine.unique())))
train_data.cuisine.unique()

# Visualize
import random
from plotly.offline import init_notebook_mode, plot

init_notebook_mode(connected=True)
import plotly.graph_objs as go

trace = go.Table(
    header=dict(values=['Cuisine', 'Number of recipes'],
                fill=dict(color=['#FFFFFF']),
                align=['left'] * 7),
    cells=dict(values=[train_data.cuisine.value_counts().index, train_data.cuisine.value_counts()],
               align=['left'] * 7))


# table banaya values color sab add kar diya
layout = go.Layout(title='Reicipies in cuisine',
                   titlefont=dict(size=20),
                   width=500, height=650,
                   paper_bgcolor='rgba(0,0,0,0)',
                   plot_bgcolor='rgba(0,0,0,0)',
                   autosize=False,
                   margin=dict(l=40, r=30, b=1, t=60, pad=1),
                   )

# table show kiya
data = [trace]
fig = dict(data=data, layout=layout)
plot(fig)
plt.show()

# table making cuisine vs recipies
label_percents = []
for i in train_data.cuisine.value_counts():
    percent = (i / sum(train_data.cuisine.value_counts())) * 100
    percent = "%.2f" % percent
    percent = str(percent + '%')
    label_percents.append(percent)


# cuisine vs recipies
data = [trace]
fig = dict(data=data, layout=layout)
plot(fig, filename='horizontal-bar')
plt.show()

print('Maximum Number of Ingredients in a Dish: ', train_data['ingredients'].str.len().max())
print('Minimum Number of Ingredients in a Dish: ', train_data['ingredients'].str.len().min())

# ingredients vs kitni recipies
trace = go.Histogram(
    x=train_data['ingredients'].str.len(),
    xbins=dict(start=0, end=80, size=1),
    marker=dict(color='#fbca5f'),
    opacity=0.75)
data = [trace]
layout = go.Layout(
    title='Dist. of length of recipy',
    xaxis=dict(title='Ingredients'),
    yaxis=dict(title='recipies'),
    bargap=0.1,
    bargroupgap=0.2)

fig = go.Figure(data=data, layout=layout)
plot(fig)
plt.show()

# ingredients vs nationalities meh kitna
labels = [i for i in train_data.cuisine.value_counts().index][::-1]
data = []
for i in range(20):
    trace = {
        "type": 'violin',
        "y": train_data[train_data['cuisine'] == labels[i]]['ingredients'].str.len(),
        "name": labels[i],
        "box": {
            "visible": True
        },
        "meanline": {
            "visible": True
        }
    }
    data.append(trace)
layout_txt = go.Layout(
    title="Recipe Length Distribution cuisine"

)

fig = go.Figure(data=data, layout=layout)
plot(fig, filename="Box Plot Styling Outliers")
plt.show()

# proc
features = []  # list of list containg the recipes
for item in train_data['ingredients']:
    features.append(item)

# Test Sample - only features - the target variable is not provided.
featurs_test = []  # list of lists containg the recipes
for item in test_data['ingredients']:
    featurs_test.append(item)

import re

featurs_processed = []  # here we will store the preprocessed training features
for item in features:
    newitem = []
    for ingr in item:
        ingr.lower()  # Case Normalization - convert all to lower case
        ingr = re.sub("[^a-zA-Z]", " ", ingr)  # Remove punctuation, digits or special characters
        ingr = re.sub((r'\b(oz|ounc|ounce|pound|lb|inch|inches|kg|to)\b'), ' ', ingr)  # Remove different units
        newitem.append(ingr)
    featurs_processed.append(newitem)

# Test
featurs_test_processed = []
for item in featurs_test:
    newitem = []
    for ingr in item:
        ingr.lower()
        ingr = re.sub("[^a-zA-Z]", " ", ingr)
        ingr = re.sub((r'\b(oz|ounc|ounce|pound|lb|inch|inches|kg|to)\b'), ' ', ingr)
        newitem.append(ingr)
    featurs_test_processed.append(newitem)

# Binary representation of the training set will be employed

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(analyzer="word",
                             ngram_range=(1, 1),  # unigrams
                             binary=True,  # count default there
                             tokenizer=None,
                             preprocessor=None,
                             stop_words=None,
                             max_df=0.99)  # >99, discard
train_X = vectorizer.fit_transform([str(i) for i in featurs_processed])
test_X = vectorizer.transform([str(i) for i in featurs_test_processed])

target_val = train_data['cuisine']

from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()
train_Y = lb.fit_transform(target_val)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, random_state=0)

# Model Making
clfs = []

from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
rfc.fit(X_train, y_train)
print('RFC LogLoss {score}'.format(score=log_loss(y_test, rfc.predict_proba(X_test))))
clfs.append(rfc)

from sklearn.svm import SVC

svc = SVC(random_state=42, probability=True, kernel='linear')
svc.fit(X_train, y_train)
print('SVC LogLoss {score}'.format(score=log_loss(y_test, svc.predict_proba(X_test))))
clfs.append(svc)