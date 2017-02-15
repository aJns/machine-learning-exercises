# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 12:09:23 2017

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

#%% 5
from sklearn.neighbors import KNeighborsClassifier
from sklearn.lda import LDA
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

classifiers = []
classifiers.append(KNeighborsClassifier())
classifiers.append(LDA())
classifiers.append(SVC())

#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

for clf in classifiers:
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print("Accuracy: " + str(score))
        
    
