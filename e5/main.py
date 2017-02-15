#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 13:02:11 2017

@author: nikulaj
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import skimage.io as imgio
import skimage.feature as sfeature
import os

#%% 4
class1Path = "./class1"
class1Files = [f for f in os.listdir(class1Path) if os.path.isfile(os.path.join(class1Path, f))]

class2Path = "./class2"
class2Files = [f for f in os.listdir(class1Path) if os.path.isfile(os.path.join(class1Path, f))]

class1Images = []
class2Images = []

for file in class1Files:
    class1Images.append(imgio.imread(class1Path + "/" + file))

for file in class2Files:
    class2Images.append(imgio.imread(class2Path + "/" + file))

#%%
colorValues = 256
R = 3
P = 8*R

featureMatrix = np.zeros( (len(class1Images) + len(class2Images), colorValues) )
labelVector = []

counter = 0
for image in class1Images:
    histogram = np.histogram(sfeature.local_binary_pattern(image, P, R), bins=colorValues)
    features = histogram[0]
    featureMatrix[counter,:] = features
    labelVector.append(0)
    counter = counter + 1
    
for image in class2Images:
    histogram = np.histogram(sfeature.local_binary_pattern(image, P, R), bins=colorValues)
    features = histogram[0]
    featureMatrix[counter,:] = features
    labelVector.append(1)
    counter = counter + 1


X = featureMatrix
y = labelVector

#%%
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import normalize
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

C_range = np.linspace(np.power(10, -5), 1)
C_range[0] += 0.01

X_norm = normalize(X)
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.20, random_state=42)


clf_list = [LogisticRegression(), SVC()]
clf_name = ["LR", "SVC"]

best_score = 0
best_C = 0
best_penalty = 0

for clf,name in zip(clf_list, clf_name):
    for C in C_range:
        for penalty in ["l1", "l2"]:
            clf.C = C
            clf.penalty = penalty
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            
            if (score > best_score):
                best_score = score
                best_C = C
                best_penalty = penalty
            
print("Best score: " + str(score) + " best C: " + str(C) + " best penalty: " + str(penalty))
            
#%% 5
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

trees = 100

classifiers = []
classifiers.append(RandomForestClassifier(n_estimators=trees))
classifiers.append(ExtraTreesClassifier(n_estimators=trees))
classifiers.append(AdaBoostClassifier(n_estimators=trees))
classifiers.append(GradientBoostingClassifier(n_estimators=trees))

#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

for clf in classifiers:
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print("Accuracy: " + str(score))














