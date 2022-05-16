#Library Imports

from flask_ngrok import run_with_ngrok
from flask import Flask

import nltk
nltk.download("stopwords")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

import nltk 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from string import punctuation
import re,string,unicodedata
import pickle


from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk import pos_tag
from nltk.corpus import wordnet

from xgboost import XGBClassifier

from collections import Counter
from imblearn.over_sampling import SMOTE

from wordcloud import WordCloud, STOPWORDS
from bs4 import BeautifulSoup

df = pd.read_csv('train.csv')

#Data Cleaning

stop = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop.update(punctuation)

def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

#Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)
# Removing URL's
def remove_between_square_brackets(text):
    return re.sub(r'http\S+', '', text)
#Removing the stopwords from text
def remove_stopwords(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop and i.strip().lower().isalpha():
            final_text.append(i.strip().lower())
    return " ".join(final_text)
#Removing the noisy text
def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    text = remove_stopwords(text)
    return text
#Apply function on review column
df['comment_text']=df['comment_text'].apply(denoise_text)

print('ORIGINAL SENTENCE :',non_toxic_comments[0])
print('-'*100)
print('CLEANED SENTENCE :',df['comment_text'][0])

#Model

# dependent and independent variable
X = df['comment_text']
y = df['toxic']

# dependent and independent variable
X = df['comment_text']
y = df['toxic']

smote = SMOTE(random_state = 402)
X_smote, Y_smote = smote.fit_resample(X,y)

sns.countplot(Y_smote)

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X_smote, Y_smote, test_size = 0.20, random_state = 0)

#Logistic Regression
lr = LogisticRegression()
#Fitting the model 
lr.fit(X_train,y_train)

# Predicting the Test set results
y_pred_lr = lr.predict(X_test)

# Accuracy, Precision,f1 and Recall
score1 = accuracy_score(y_test,y_pred_lr)
score2 = precision_score(y_test,y_pred_lr)
score3= recall_score(y_test,y_pred_lr)
score4 = f1_score(y_test,y_pred_lr)
print("---- Scores ----")
print("Accuracy score is: {}%".format(round(score1*100,2)))
print("Precision score is: {}".format(round(score2,2)))
print("Recall score is: {}".format(round(score3,2)))
print("F1 Score score is: {}".format(round(score4,2)))

#Naive Bayes
# Fitting Naive Bayes to the Training set
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_nb = classifier.predict(X_test)

# Accuracy, Precision,f1 and Recall
score1 = accuracy_score(y_test,y_pred_nb)
score2 = precision_score(y_test,y_pred_nb)
score3 = recall_score(y_test,y_pred_nb)
score4 = f1_score(y_test,y_pred_nb)
print("---- Scores ----")
print("Accuracy score is: {}%".format(round(score1*100,2)))
print("Precision score is: {}".format(round(score2,2)))
print("Recall score is: {}".format(round(score3,2)))
print("F1 Score score is: {}".format(round(score4,2)))

#XGBClassifier
clf = XGBClassifier()
clf.fit(X_train, y_train)

# Predicting the Test set results
y_pred_xg = classifier.predict(X_test)

# Accuracy, Precision,f1 and Recall
score1 = accuracy_score(y_test,y_pred_xg)
score2 = precision_score(y_test,y_pred_xg)
score3= recall_score(y_test,y_pred_xg)
score4 = f1_score(y_test,y_pred_nb)
print("---- Scores ----")
print("Accuracy score is: {}%".format(round(score1*100,2)))
print("Precision score is: {}".format(round(score2,2)))
print("Recall score is: {}".format(round(score3,2)))
print("F1 Score score is: {}".format(round(score4,2)))

#Saving the Model
file = open('toxic_comments.pkl', 'wb')

# dump information to that file
pickle.dump(clf, file)
pickle.dump(cv, open('transform.pkl', 'wb'))




