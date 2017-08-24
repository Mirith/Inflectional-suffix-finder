# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 10:16:58 2017

@author: Mirith

Info: Predicts English inflectional suffixes using machine learning, and saves the trained model for later use (so you can use it on other datasets without training it all over again)

"""

###############################################
# all imports... 

# to read dataset
import pandas as pd
# to split it into training and testing
from sklearn.model_selection import train_test_split
# for machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
# also machine learning, but not an actual algorithm
from sklearn.pipeline import Pipeline
# to see how well the training went
from sklearn.metrics import accuracy_score
# to dump the trained analyzer into a .pkl file
from sklearn.externals import joblib

# directory where data is stored
dir = "C:/Users/________/Documents/engsuffixes.txt"

# reading the dataset into something computers can work with
# basically this means that each entry has multiple points of data, # and each point is delimited by a space
# can be things like backslashes or hyphens or commas too... 
dataset = pd.read_csv(dir, header = 0, delimiter = " ")

###############################################
# splitting data into training and testing sets
#
# docs_train -- training input data
# docs_test -- testing input data
# class_train -- training input results
# class_test -- testing input results
docs_train, docs_test, class_train, class_test = train_test_split(dataset["a.1"], dataset["#"], test_size = 0.20, random_state = 48)
###############################################
# machine learning thingies!

# analyzes by character, ranging in chunks of 1 to four characters
cv = CountVectorizer(analyzer = 'char', ngram_range = (1,4))

# looks at term frequency in dataset, not totally necessary
# does improve accuracy by about 1% though
tf = TfidfTransformer()

# not even sure prof just said do it
lr = LogisticRegression(C = 6.5)

# nice pipeline that does all the things in one go
morph = Pipeline([('vect', cv), ('tfidf', tf), ('clf', lr)])

# training
morph = morph.fit(docs_train, class_train)

# testing
predicted = morph.predict(docs_test)
print('predicted accuracy:', accuracy_score(class_test, predicted) * 100, '%')

###############################################
# result:
#
# predicted accuracy: 93.2880728283 %

###############################################
# saving the trained model
dump_dir = "C:/Users/_______/Documents/morph.pkl"
joblib.dump(morph, dump_dir)