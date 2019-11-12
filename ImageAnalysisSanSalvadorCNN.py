"""
This code explores Different Models of Convolutional Neural Networks
for the San Salvador Gang Project
@author: falba and ftop
"""
import os
import google_streetview.api
import pandas as pd
import numpy as np
import sys

import matplotlib.image as mp_img
from matplotlib import pyplot as plot
from skimage import io
from skimage.color import rgb2gray
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import train_test_split

#pip install tensorflow keras numpy skimage matplotlib
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras import layers
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.utils import to_categorical
from keras.layers import  LSTM, Embedding
from keras.preprocessing.image import ImageDataGenerator
from sklearn.linear_model import LogisticRegression
from numpy import where
from keras import regularizers
import random

#### Load the data
## Set Directory
os.chdir('C:/Users/falba/Dropbox/ImageAnalysis/San Salvador/GangBoundaries')
df = pd.read_csv("C:/Users/falba/Dropbox/ImageAnalysis/San Salvador/GangBoundaries/sample.csv", header=0)
df

astr = "C:/Users/falba/Dropbox/ImageAnalysis/San Salvador/GangBoundaries/"

# Let's create a sample for testing and training:
train, test = train_test_split(df, test_size=0.25, random_state=38)

# Obtaining the image data of testing 
test_cases=[]
test_class=[]

file=test.file

for x in file:
    image = io.imread(astr+x)
    image =rgb2gray(image)
    test_cases.append(image)

test_cases=np.reshape(np.ravel(test_cases),(579,480,480,-1))

for index, Series in test.iterrows():
    test_class.append(Series["gang_territory10"])
    
test_class=np.reshape(test_class,(579,-1))

# The image data of training 
train_cases=[]
train_class=[]

fileT=train.file

for x in fileT:
    image = io.imread(astr+x)
    image=rgb2gray(image)
    train_cases.append(image)
    
train_cases=np.reshape(train_cases,(1735,480,480,-1))

for index, series in train.iterrows():
    train_class.append(series["gang_territory10"])

train_class=np.reshape(train_class,(1735,-1))

## To Categorical 
#y_train = to_categorical(train_class)
#y_test= to_categorical(test_class)

input_dim = train_cases.shape[1]
maxlen = 100

### Now let's try a Convolutional Neural Networks
# Seeting up Convolution Layers and Filters

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(480,480,1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])
model.summary()
hist_1 = model.fit(train_cases,train_class,verbose=False,epochs=50,validation_data=(test_cases,test_class),batch_size=10)
hist1acc=model.evaluate(test_cases,test_class)
#accuracy: 0.6165
# plot loss during training
plot.subplot(211)
#plot.title('Loss / Binary Crossentropy')
plot.plot(hist_1.history['loss'], label='Train')
plot.plot(hist_1.history['val_loss'], label='Test')
plot.legend()
plot.show()

plot.subplot(212)
#plot.title('Accuracy / Binary Crossentropy')
plot.plot(hist_1.history['accuracy'], label='Train')
plot.plot(hist_1.history['val_accuracy'], label='Test')
plot.legend()
plot.show()
#plot.savefig('LossBinCross.png')


# Binary CrossEntropy - Model 2
model = Sequential()
model.add(Conv2D(8, (5, 5), activation='relu', input_shape=(480,480,1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(2, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
hist_2 = model.fit(train_cases,train_class,verbose=False,epochs=30,validation_data=(test_cases,test_class),batch_size=10)
# evaluate the model
hist2acc=model.evaluate(test_cases,test_class)
#Accuracy 0.5354

# plot accuracy during training
plot.subplot(212)
plot.title('Accuracy / Binary Crossentropy')
plot.plot(hist_2.history['accuracy'], label='Train')
plot.plot(hist_2.history['val_accuracy'], label='Test')
plot.legend()
plot.show()

plot.subplot(211)
plot.title('Loss / Binary Crossentropy')
plot.plot(hist_2.history['loss'], label='Train')
plot.plot(hist_2.history['val_loss'], label='Test')
plot.legend()
plot.show()

## Seems like EPOCH migh be too high. Optimal can be less than 10
## Maybe because overfitting


# Binary CrossEntropy - Model 3
model = Sequential()
model.add(Conv2D(10, (11, 11), activation='relu', input_shape=(480,480,1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(20, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D(100, (4, 4), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
hist_3 = model.fit(train_cases,train_class,verbose=False,epochs=30,validation_data=(test_cases,test_class),batch_size=10)
# evaluate the model
hist3acc=model.evaluate(test_cases,test_class)
#Accuracy 53%


## Graphs 
plot.subplot(212)
#plot.title('Accuracy / Binary Crossentropy')
plot.plot(hist_3.history['accuracy'], label='Train')
plot.plot(hist_3.history['val_accuracy'], label='Test')
plot.legend()
plot.show()

plot.subplot(211)
#plot.title('Loss / Binary Crossentropy')
plot.plot(hist_3.history['loss'], label='Train')
plot.plot(hist_3.history['val_loss'], label='Test')
plot.legend()
plot.show()


## LET'S TRY REGULARIZATION 
model = Sequential()
model.add(Conv2D(8, (5, 5), activation='relu', input_shape=(480,480,1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(2, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Flatten())
model.add(Dense(10,
                kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))
#model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
hist= model.fit(train_cases,train_class,verbose=False,epochs=50,validation_data=(test_cases,test_class),batch_size=10)
# evaluate the model
hist_acc=model.evaluate(test_cases,test_class)
#Accuracy 53% 

# plot accuracy during training
plot.subplot(212)
plot.title('Accuracy / Binary Crossentropy')
plot.plot(hist.history['accuracy'], label='Train')
plot.plot(hist.history['val_accuracy'], label='Test')
plot.legend()
plot.show()

plot.subplot(211)
plot.title('Loss / Binary Crossentropy')
plot.plot(hist.history['loss'], label='Train')
plot.plot(hist.history['val_loss'], label='Test')
plot.legend()
plot.show()

## It didnt help much with accuracy but it did with the loss



