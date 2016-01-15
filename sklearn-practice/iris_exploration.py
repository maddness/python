# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 11:43:24 2016

@author: aostrikov
"""

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.datasets import load_iris
from sklearn import svm


data = load_iris()

data['feature_names']
data['data']
data['target']
data['target_names']

# ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

# iris_type = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}

df = pd.DataFrame(data['data'], columns = ['sl', 'sw', 'pl', 'pw'])
df['type'] = data['target']

train, test = train_test_split(df, test_size = 0.2)

X = train[['sl', 'sw', 'pl', 'pw']]
#X = train[['sl', 'sw']]
y = train['type']

svc = svm.SVC(kernel='rbf', C=1.0)
svc.fit(X, y)

X_test = test[['sl', 'sw', 'pl', 'pw']]
#X_test = test[['sl', 'sw', 'pl', 'pw']]

predicted = svc.predict(X_test)
actual = test['type']

print confusion_matrix(predicted, actual)

#plt.figure(figsize=(8, 8))

#plt.subplots_adjust(bottom=.05, top=.9, left=.05, right=.95)
plt.scatter(train['sl'], train['sw'], marker='x')
plt.scatter(test['sl'], test['sw'], marker='o')
#plt.show()







