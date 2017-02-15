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

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation, MaxPooling2D, Flatten
from keras.layers.convolutional import Convolution2D
from sklearn.cross_validation import train_test_split

# %% 4
class1Path = "./class1"
class1Files = [f for f in os.listdir(class1Path) if os.path.isfile(os.path.join(class1Path, f))]

class2Path = "./class2"
class2Files = [f for f in os.listdir(class1Path) if os.path.isfile(os.path.join(class2Path, f))]

class1Images = []
class2Images = []

for file in class1Files:
    class1Images.append(imgio.imread(class1Path + "/" + file))

for file in class2Files:
    class2Images.append(imgio.imread(class2Path + "/" + file))

# %%
imageCount = len(class1Images) + len(class2Images)
channels = 3
width = 64
height = 64
featureMatrixShape = (imageCount, channels, width, height)

featureMatrix = np.zeros(featureMatrixShape)
labelVector = []

counter = 0
for image in class1Images:
    featureMatrix[counter, :] = np.transpose(image)
    labelVector.append(0)
    counter = counter + 1


for image in class2Images:
    featureMatrix[counter, :] = np.transpose(image)
    labelVector.append(1)
    counter = counter + 1


X = featureMatrix.astype(float)
y = labelVector

# %%

X = (X - np.min(X))/np.max(X)

new_y = np.zeros((len(y), 2))
new_y[np.arange(len(y)), y] = 1

X_train, X_test, y_train, y_test = train_test_split(X, new_y, test_size=0.20, random_state=42)

# %%

N = 10
# Number of feature maps
w, h = 3, 3
# Conv. window size
model = Sequential()

model.add(Convolution2D(nb_filter=N,
                        nb_col=w,
                        nb_row=h,
                        activation='relu',
                        border_mode='same',
                        input_shape=(3, 64, 64)))

model.add(MaxPooling2D((2, 2)))

model.add(Convolution2D(nb_filter=N,
                        nb_col=w,
                        nb_row=h,
                        border_mode='same',
                        activation='relu'))

model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(2, activation='sigmoid'))

# %%

model.compile(optimizer='SGD',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, nb_epoch=20, batch_size=32, validation_data=[X_test, y_test])

score = model.evaluate(X_test, y_test, batch_size=32)
print("Got a final score of " + str(score))

