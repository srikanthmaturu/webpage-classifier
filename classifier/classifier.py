#!/usr/bin/env python

from time import time
import sys
import os
import re
from glob import glob

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_mlcomp

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import BernoulliRBM
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.pipeline import Pipeline

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def classify(directory):
        result = [(x,y) for x in os.walk(directory) for y in glob(os.path.join(x[0], '*.txt'))]
        temp = [ x[1] for x in os.walk(directory)]
        y_train_names = temp[0]
        print "Categories: ", y_train_names

        #print result
        ls = []
        y_train = []

        for (cat,page) in result:
            # if cat[0].split('/')[-1] == 'texas':
                    fp = open(page,"rb")
                    cleaned_data = fp.read()
                    fp.close()
                    ls.append(cleaned_data)
                    # y_train.append(cat[0].split('/')[-2])
                    y_train.append(y_train_names.index(cat[0].split('/')[-2])+1)

        print "No of Pages : ", len(y_train)
        y_train = np.asarray(y_train)
        vectorizer = TfidfVectorizer(token_pattern=r'\b[a-z]{2,}\b', max_df=0.3, min_df=2, sublinear_tf=True)
        #print "Preparing vectorizer"
        X_train = vectorizer.fit_transform(ls)
        clf = MultinomialNB().fit(X_train, y_train)
        X_test = vectorizer.fit_transform(ls)
        predicted = clf.predict(X_test)
        # for page, category in zip(y_train,predicted)
        #    print('%s => %s' %(page,category)

        accuracy = np.mean(y_train == predicted)
        print "Accuracy of Classifier: %f ", accuracy

        print "Classifier Report: "
        y_train_names = np.asarray(y_train_names)
        #print metrics.classification_report(y_train, predicted, y_train_names)

        print "Confusion Matrix : "
        print metrics.confusion_matrix(y_train, predicted)


if __name__ == '__main__':
        args = sys.argv[1:]
        if not args:
            print >> sys.stderr, 'SYNTAX: classifier.py [directory]'
            sys.exit(-1)

        classify(args[0])






