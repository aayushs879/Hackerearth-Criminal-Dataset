# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 14:05:59 2018

@author: aayush
"""
#Importin Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing Dataset
dataset = pd.read_csv('criminal_train.csv')
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

#Dropping totally irrelevant features
X = np.delete(arr = X,obj =[-10,-17,-28,-29,-33,-54,-56,-61,-59,-62,-69], axis = 1)

#converting independent variables' values to positive 
x1=X
x2 = np.ones(shape = (np.size(X[:,1]),np.size(X[1,:]))).astype(int)
X = np.add(x1 , x2 ).astype(int)

#feature selection on the basis of chi squared test
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
select = SelectKBest(chi2, 42)
sel = select.fit(X,y)
feature_score = sel.scores_  #visualization of features' scores on the basis of chi2
X = sel.transform(X)

#for cross validation
from sklearn.cross_validation import train_test_split as tts
X_train, X_test, y_train, y_test = tts(X,y,test_size = 2000,random_state = 1)


#feature scaling
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
scale.fit(X_train)
X_train = scale.transform(X_train)
X_test = scale.transform(X_test)

#while training on whole dataset, trained the whole dataset on the performance of svc
scale2 = StandardScaler()
scale2.fit(X)
X = scale2.transform(X)

#testing score of multi layered perceptron
from sklearn.neural_network import MLPClassifier as mlp
from sklearn.model_selection import learning_curve
cv_score = learning_curve(mlp(activation = 'logistic', hidden_layer_sizes = (25,1), solver = 'lbfgs', alpha = .01, max_iter= 400),X_train,y_train )

#training score of multi layered perceptron
classifier = mlp(activation = 'logistic', hidden_layer_sizes = (50,6), solver = 'lbfgs', alpha = .1, max_iter= 400)
classifier.fit(X_train,y_train)

#evaluation of mlp
from sklearn.metrics import confusion_matrix as cmm
cmm(y_test, classifier.predict(X_test))

from sklearn.metrics import accuracy_score
a = accuracy_score(y_test, classifier.predict(X_test))


#implementi support vector classifier
from sklearn.svm import SVC
classifier1 = SVC(kernel = 'rbf', max_iter = 2000, C = .03)
classifier1.fit(X_train,y_train)

#evaluating svc
from sklearn.metrics import confusion_matrix as cmm
cmm(y_test, classifier1.predict(X_test))

from sklearn import metrics
metrics.f1_score(y_test, classifier1.predict(X_test))

#training classifier on the whole dataset
classifier1.fit(X,y)

#predictions on test set

test_set = pd.read_csv('criminal_test.csv')
test = test_set.iloc[:,1:].values
test = sel.transform(test)
test = scale2.transform(test)

predictions = classifier1.predict(test)
df = pd.DataFrame(predictions)

#first column belonged to person id, submissions made on the prescribed format of submission script provided by hackerearth
submission = pd.concat([test_set.iloc[:,0],df], axis = 1)

#final submission made by training classifier on whole dataset

pd.DataFrame.to_csv(self = submission, path_or_buf = 'C:\Aayush\Machine Learning\submissions2.csv') 