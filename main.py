#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 12:26:06 2018

@author: alok
"""

# importing libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# importing dataset
dataset = pd.read_csv('./input/train.csv')
test = pd.read_csv('./input/test.csv')

classifier = DecisionTreeClassifier()

# training dataset
X_train = dataset.iloc[:, 1:786].values
Y_train = dataset.iloc[:,0].values

classifier.fit(X_train, Y_train)

# testing datatset
y_pred = classifier.predict(test)
temp = [i for i in range(1, 28001)]

submission = pd.DataFrame({
        "ImageId" : temp,
        "Label" : y_pred
        })

submission.to_csv('digit_recognizer.csv', index=False)
