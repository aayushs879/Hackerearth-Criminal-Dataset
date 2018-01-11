# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 14:05:59 2018

@author: aayush
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('criminal_train.csv')

X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values
X = np.delete(arr = X,obj =[-10,-17,-28,-29,-33,-54,-56,-61,-59,-62,-69], axis = 1)

x1=X
x2 = np.ones(shape = (np.size(X[:,1]),np.size(X[1,:]))).astype(int)

X = np.add(x1 , x2 ).astype(int)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
select = SelectKBest(chi2, 42)
sel = select.fit(X,y)
feature_score = sel.scores_
X = sel.transform(X)

from sklearn.cross_validation import train_test_split as tts
X_train, X_test, y_train, y_test = tts(X,y,test_size = 2000,random_state = 1)

from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
scale.fit(X_train)
X_train = scale.transform(X_train)
X_test = scale.transform(X_test)

from sklearn.neural_network import MLPClassifier as mlp

from sklearn.model_selection import learning_curve
cv_score = learning_curve(mlp(activation = 'logistic', hidden_layer_sizes = (25,1), solver = 'lbfgs', alpha = .01, max_iter= 400),X_train,y_train )


classifier = mlp(activation = 'logistic', hidden_layer_sizes = (50,6), solver = 'lbfgs', alpha = .1, max_iter= 400)
classifier.fit(X_train,y_train)

from sklearn.metrics import confusion_matrix as cmm
cmm(y_test, classifier1.predict(X_test))

from sklearn.metrics import accuracy_score
a = accuracy_score(y_test, classifier1.predict(X_test))

from sklearn.svm import SVC
classifier1 = SVC(kernel = 'rbf', max_iter = 2000, C = .03)
classifier1.fit(X_train,y_train)

from sklearn import metrics
metrics.f1_score(y_test, classifier1.predict(X_test))


test_set = pd.read_csv('criminal_test.csv')
test = test_set.iloc[:,1:].values
test = sel.transform(test)
test = scale.transform(test)

predictions = classifier1.predict(test)
df = pd.DataFrame(predictions)


submission = pd.concat([test_set.iloc[:,0],df], axis = 1)

pd.DataFrame.to_csv(self = submission, path_or_buf = 'C:\Aayush\Machine Learning\submissions2.csv') 


plt.hist(X[:,5])


from sklearn import naive_bayes
clf  = naive_bayes.GaussianNB()
clf.fit(X_train, y_train)
from sklearn.metrics import confusion_matrix as cmm
cmm(y_test, clf.predict(X_test))