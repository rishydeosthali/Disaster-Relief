# Import Python tools for loading/navigating data
from collections import Counter
from random import sample
from importlib.machinery import SourceFileLoader
import re
import numpy as np
from os.path import join
import warnings
warnings.filterwarnings("ignore")

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import *
from nltk.corpus import stopwords
nltk.download('stopwords' ,quiet=True)
from sklearn.model_selection import train_test_split
import pandas as pd
from TweetModels import *

# Load the data.
disaster_tweets = pd.read_csv('disaster_data.csv',encoding ="ISO-8859-1")

#Read the tweet data and convert it to lowercase
tweets = disaster_tweets['text'].str.lower() 
tweets = tweets.apply(lambda x: re.sub(r'[^a-zA-Z0-9]+', ' ',x))

#Extract the labels from the csv
tweet_labels = disaster_tweets['category']

# Tokenize all the tweets and remove stopwords
tokenized_tweets = [word_tokenize(t) for t in tweets]
tweet_set = remove_stopwords(tokenized_tweets)

#Split the Data into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(tweet_set, tweet_labels, test_size=0.2, random_state=1)

# Train the model
model, train_countvect = train_model(X_train, y_train)

#Predict labels for test set
y_pred = predict (X_test, train_countvect, model)
print()
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print()

# Display evaluated tweets vs original category
table=pd.DataFrame([[" ".join(t) for t in X_test],y_pred, y_test]).transpose()
table.columns = ['Tweet', 'Predicted Category', 'True Category']

# Show Confusion Matrix
plot_confusion_matrix(y_test,y_pred)
