from flask_ngrok import run_with_ngrok
from flask import Flask
from flask import Flask,render_template,url_for,request
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import pickle

#Load the model
filename = 'toxic_comments.pkl'
clf = pickle.load(open(filename, 'rb'))
cv = pickle.load(open('transform.pkl','rb'))
app = Flask(__name__)
run_with_ngrok(app)

@app.route('/')
def home():
  return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
  if request.method== 'POST': 
    data = [message]
    vect = cv.transform(data).toarray()
    my_prediction = clf.predict(vect)
  return render_template('home.html', prediction = my_prediction)

if __name__=='main':
  app.run()

#run_with_ngrok(app)

