from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import multilabel_confusion_matrix
from sklearn import metrics
import pandas as pd

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import *
from nltk.corpus import stopwords
nltk.download('stopwords' ,quiet=True)
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

def train_model(tweets_to_train,train_labels):
  """
  param: tweets_to_train - list of tweets to train on
  return: the vectorizer, the logistic regression model, the train_vector
  """
  
  train_tweets = [" ".join(t) for t in tweets_to_train]
  train_tweets_label = [l for l in train_labels]
  
  ### Your code starts here ###

  vectorizer = CountVectorizer() # initialize CountVectorizer
  train_vect = vectorizer.fit_transform(train_tweets)

  model = LogisticRegression() # create LogisticRegression Model
  model.fit(train_vect, train_labels) # train on transformed train data and our labels

  ### Your code ends here ###
  
  return model, vectorizer
  
def predict(tweets_to_test, vectorizer, model):
  """
  param: tweets_to_test - list of tweets to test the model on
  param: vectorizer - the CountVectorizer
  param: model - the LogisticRegression model
  return result (the prediction), the test_vect
  """
  
  test_tweets = [" ".join(t) for t in  tweets_to_test]
  #print(test_tweets)

  ### Your code starts here

  test_vect = vectorizer.transform(test_tweets) # Use .transform to vectorize our tweets
  result = model.predict(test_vect) # Have your model predict on the vectorized tweets

  ### Your code ends here

  return result
  
#@title Run this to load the helper function for plotting our confusion matrix!
'''
Plots the confusion Matrix and saves it
'''
def plot_confusion_matrix(y_true,y_predicted):
  cm = metrics.confusion_matrix(y_true, y_predicted)
  print ("Plotting the Confusion Matrix")
  labels = ['Energy', 'Food', 'Medical', 'None', 'Water']
  df_cm = pd.DataFrame(cm,index =labels,columns = labels)
  fig = plt.figure()
  res = sns.heatmap(df_cm, annot=True,cmap='Blues', fmt='g')
  plt.yticks([0.5,1.5,2.5,3.5,4.5], labels,va='center')
  plt.title('Confusion Matrix - TestData')
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.show()
  plt.close()

stopword_set = set(stopwords.words('english'))

# Complete the following function to remove the stopwords from the tokenized tweets 
def remove_stopwords(token_list):
  filtered_sentences = [] 
  ### YOUR CODE HERE
  for tweet in token_list:
    new_tweet = []
    for word in tweet:
      if word not in stopword_set:
        new_tweet.append(word)
    filtered_sentences.append(new_tweet)
  ### END CODE
  return filtered_sentences
  
