import nltk
nltk.download("stopwords")
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
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

#Retrieving data
df = pd.read_csv('train.csv')
df.head()