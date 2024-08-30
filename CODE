// using dataset from https://www.kaggle.com/c/fake-news/data?select=train.csv
//id: unique id for a news article
//title: the title of a news article
//author: author of the news article
//text: the text of the article; could be incomplete
//label: a label that marks whether the news article is real or fake
//1: Fake news
//0: real News
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
nltk.download('stopwords')
# loading the dataset to a pandas DataFrame
news_dataset = pd.read_csv('/content/train.csv')
news_dataset.isnull().sum()
# replacing the null values with empty string
news_dataset = news_dataset.fillna('')
# merging the author name and news title
news_dataset['content'] = news_dataset['author']+' '+news_dataset['title']
# separating the data & label
X = news_dataset.drop(columns='label', axis=1)
Y = news_dataset['label']
port_stem = PorterStemmer()
news_dataset['content'] = news_dataset['content'].apply(stemming)
#separating the data and label
X = news_dataset['content'].values
Y = news_dataset['label'].values
# converting the textual data to numerical data
vectorizer = TfidfVectorizer()
vectorizer.fit(X)

X = vectorizer.transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify=Y, random_state=2)
//training the dataset
model = LogisticRegression()
model.fit(X_train, Y_train)
# accuracy score on the training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of the training data : ', training_data_accuracy)
print('Accuracy score of the test data : ', test_data_accuracy)
X_new = X_test[3]

prediction = model.predict(X_new)
print(prediction)

if (prediction[0]==0):
  print('The news is Real')
else:
  print('The news is Fake')
