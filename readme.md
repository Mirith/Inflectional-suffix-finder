**Last updated on August 24th, 2017** by [Mirith](https://github.com/Mirith)

# Overview

This project uses a dataset of English words and machine learning to derive inflectional suffixes for those English words.  Accuracy is about 93 percent.  It uses [pandas](http://pandas.pydata.org/pandas-docs/stable/io.html) to read the data and [sklearn/scikitlearn](http://scikit-learn.org/stable/modules/classes.html) to process it.  

# Usage

You'll need python (this was written in python 3 on your computer as well as the dataset.  I've only included all words beginning with 'a', as I'm unsure if my professor wants the data online.  With the full dataset, training takes over an hour.  The smaller one should take far less time.  

# Files

## engsuffixes.txt

This is the dataset.  Each row is another entry.  Each entry has three fields -- the first is the unaffixed word, the second is the affixed (if any) word, and the third is the affix itself.  If there is no affix present, ie the first and second fields match, then the third field is a hash.  

> affix affix #
>
> affix affixes es

Like so.  

## inflectional suffixes.py

The code itself.  Hopefully it's well enough commented to make sense, but in plain English the code does a few things.  It reads the data from the text file, splits it into training and testing sets, trains a machine learning model, and then predicts the accuracy of the model.  

## morph.pkl

A dump of the trained model.  This was trained with the full dataset.  You can load it with the [load function of sklearn's joblib](https://pythonhosted.org/joblib/generated/joblib.load.html#joblib.load).  

