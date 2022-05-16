import ssl
import certifi
from urllib.request import urlopen
request = "https://example.com"
urlopen(request, context=ssl.create_default_context(cafile=certifi.where()))

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

''' 
!!! In this project, we are going to focus on classifying user comments as either 'toxic' or 'non-toxic'
Note, there is innapropriate language and derogatory comments as part of the dataset, so please be aware. 
'''
#Retrieving data
df = pd.read_csv('train.csv')
df.head()

# We only need the toxic and non-toxic comments, so drop other features
df.drop(['id','severe_toxic','obscene','threat','insult','identity_hate'],axis=1,inplace=True)

df.shape

sns.countplot(df['toxic'])

df['toxic'].value_counts()

df['Number_of_words'] = df['comment_text'].apply(lambda x:len(str(x).split()))
df.head()

df.describe()

#Note, we have 159,751 words with the longest sentence of 1411 and an average of 67 words per comment
# There is a large deviation of 99, indicating a wide spread

print('Number of sentences having one word are',len(df[df['Number_of_words']==1]))

df[df['Number_of_words']==1]['comment_text']

#Frequency Distribution of Number of Words for each text extracted
plt.style.use('ggplot')
plt.figure(figsize=(12,6))
sns.distplot(df['Number_of_words'],kde = False,color="red",bins=200)
plt.title("Frequency distribution of number of words for each text extracted", size=20)

# Comparison of toxic and non-toxic comments
toxic_comments = df[df['toxic'] ==1]['comment_text']
toxic_comments.reset_index(inplace=True,drop=True)
for i in range(5):
    print(toxic_comments[i])

non_toxic_comments = df[df['toxic'] ==0]['comment_text']
non_toxic_comments.reset_index(inplace=True,drop=True)
for i in range(5):
    print(non_toxic_comments[i])

#Bar Plot of Characters in Sentence

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,8))
text_len=df[df['toxic']==1]['comment_text'].str.len()
ax1.hist(text_len,color='red')
ax1.set_title('Toxic Comment')
text_len=df[df['toxic']==0]['comment_text'].str.len()
ax2.hist(text_len,color='blue')
ax2.set_title('Non-Toxic Commet')
fig.suptitle('Characters in Sentence')
plt.show()

#Bar Plot of Number of Words In Each Text
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,8))
text_len=df[df['toxic']==1]['comment_text'].str.split().map(lambda x: len(x))
ax1.hist(text_len,color='green')
ax1.set_title('Toxic Comments')
text_len=df[df['toxic']==0]['comment_text'].str.split().map(lambda x: len(x))
ax2.hist(text_len,color='yellow')
ax2.set_title('Non-Toxic Comment')
fig.suptitle('Words in Sentence')
plt.show()

#Toxic Comments examined
toxic_text = ' '.join(df.loc[df.toxic == 1, 'comment_text'].values)
toxic_text_trigrams = [i for i in ngrams(toxic_text.split(), 3)]
Counter(toxic_text_trigrams).most_common(30)

#Non-toxic comments examined
non_toxic_text = ' '.join(df.loc[df.toxic == 0, 'comment_text'].values)
non_toxic_text_trigrams = [i for i in ngrams(non_toxic_text.split(), 3)]
Counter(non_toxic_text_trigrams).most_common(30)

# Word Cloud of Toxic and Non-toxic Comments
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[20, 5])
wordcloud1 = WordCloud( background_color='white',
                        width=600,
                        height=400).generate(" ".join(toxic_comments))
ax1.imshow(wordcloud1)
ax1.axis('off')
ax1.set_title('Toxic Comments',fontsize=40);

wordcloud2 = WordCloud( background_color='white',
                        width=600,
                        height=400).generate(" ".join(non_toxic_comments))
ax2.imshow(wordcloud2)
ax2.axis('off')
ax2.set_title('Non Toxic Comments',fontsize=40);


