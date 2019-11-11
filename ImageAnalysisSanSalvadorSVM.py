#This Code will run some image analysis of data extracted from Google Earth Images-El Salvador City*
## Author: Fabiola Alba Vivar and Furkan Top
## October 2019

## Importing Packages:
# pip install google-api-python-client
# pip install earthengine-api
# pip install cryptography
# pip install tools
# pip install google_streetview
#pip install google_streetview

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
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn import metrics

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras import layers
## Set Directory
os.chdir('C:/Users/falba/Dropbox/ImageAnalysis/San Salvador/GangBoundaries')
df = pd.read_csv("C:/Users/falba/Dropbox/ImageAnalysis/San Salvador/GangBoundaries/sample.csv", header=0)
df

astr = "C:/Users/falba/Dropbox/ImageAnalysis/San Salvador/GangBoundaries/"

###############################################################
## First: let's look at the data
## Plot the images to see how different Gang and No Gang Area look

# No Gang Area:
image_NG = io.imread(astr+"GE_2176.png")
# Gang Area:
image_GA = io.imread(astr+"GE_2167.png")

plot.imshow(image_NG)
plot.imshow(image_GA)

NG_gray = rgb2gray(image_NG)
GA_gray = rgb2gray(image_GA)

plot.imshow(NG_gray)
plot.colorbar()

plot.imshow(GA_gray)
plot.colorbar()

plot.hist(NG_gray.ravel(), fc='k', ec='k')
plot.hist(GA_gray.ravel(), fc='k', ec='k')

#Note: No gang areas might look darker: more trees (?)
##############################################################

# Now:Let's split the data  Test/Train 
train, test = train_test_split(df, test_size=0.2, random_state=38)

# Normalizing the image data of testing 
test_cases=[]
test_class=[]

file=test.file

for x in file:
    image = io.imread(astr+x)
    test_cases.append(np.reshape(rgb2gray(image),-1))

mi_standard_scaler = StandardScaler()
mi_standard_scaler.fit(test_cases)
scaled_test = mi_standard_scaler.transform(test_cases)

for index, Series in test.iterrows():
    test_class.append(Series["gang_territory10"])

# Normalizing the image data of training 
train_cases=[]
train_class=[]

fileT=train.file

for x in fileT:
    image = io.imread(astr+x)
    train_cases.append(np.reshape(rgb2gray(image),-1))
    
se_standard_scaler = StandardScaler()
se_standard_scaler.fit(train_cases)
scaled_train = se_standard_scaler.transform(train_cases)

for index, series in train.iterrows():
    train_class.append(series["gang_territory10"])

############################################################
############################################################
# Now, let's run some models:
 ############################################################
############################################################
   
## Testing Models for Binary Classification: 
## Simple Logistic Regression
clf = LogisticRegression()
log_reg= clf.fit(train_cases,train_class)
log_pred = clf.predict(test_cases)

#Accurancy
# In train:100%
# In test:  0.% 
log_train_acc = clf.score(train_cases, train_class)
log_test_acc = clf.score(test_cases, test_class)

cnf_matrix = metrics.confusion_matrix(test_class, log_pred)
cnf_matrix


## Now let's try SVM with different parameters

# Using default parameters in rbf kernel:
predDef=[]
mi_SVM = svm.SVC(gamma='scale')
mi_SVM.fit(train_cases,train_class)
mi_SVM.get_params
for elem in test_cases:
    predDef.append(mi_SVM.predict([elem]))
print(predDef)


## Accuracy :
# Train: 0.45
# Test: 0.61
accu_miSVM_train=accuracy_score(mi_SVM.predict(scaled_train),train_class)
accu_miSVM_test=accuracy_score(predDef,test_class)

# A linear kernel:
predLin=[]
li_SVM = svm.SVC(gamma='scale',kernel='linear')
li_SVM.fit(scaled_train,train_class)
li_SVM.get_params
for elem in test_cases:
    predLin.append(li_SVM.predict([elem]))
print(predLin)

## Accuracy :
# Train: 1
# Test: 0.54
accu_liSVM_train=accuracy_score(li_SVM.predict(scaled_train),train_class)
accu_liSVM_test=accuracy_score(predLin,test_class)

# A non linear kernel:
predPol=[]
Pol_SVM = svm.SVC(gamma='scale',kernel= 'poly')
Pol_SVM.fit(scaled_train,train_class)
Pol_SVM.get_params
for elem in test_cases:
    predPol.append(Pol_SVM.predict([elem]))
print(predPol)

## Accuracy :
# Train: 0.99
# Test: 0.54
accu_poSVM_train=accuracy_score(Pol_SVM.predict(scaled_train),train_class)
accu_poSVM_test=accuracy_score(predPol,test_class)


# Naive Bayesian with Gaussian Distribution:
predGau=[]
clf = GaussianNB()
clf.fit(scaled_train,train_class)
GaussianNB(priors=None, var_smoothing=1e-09)
for elem in test_cases:
   predGau.append(clf.predict([elem]))
print(predGau)

## Accuracy :
# Train: 0.64
# Test: 0.54
accu_gau_train=accuracy_score(clf.predict(scaled_train),train_class)
accu_gau_test=accuracy_score(predGau,test_class)

##### Analyzing Results:

## Accuracy Train
objects = ('RBF', 'Linear', 'Poly', 'Naive Gaussian')
y_pos = np.arange(len(objects))
performance = [0.45,1,0.99,0.64 ]

plot.bar(y_pos, performance, align='center', alpha=0.5)
plot.xticks(y_pos, objects)
plot.ylabel('Train Accuracy')
plot.title('Kernel Type')

plot.show()

## Accuracy Test
objects = ('RBF', 'Linear', 'Poly', 'Naive Gaussian')
y_pos = np.arange(len(objects))
performance = [0.61,0.54,0.54,0.54 ]

plot.bar(y_pos, performance, align='center', alpha=0.5)
plot.xticks(y_pos, objects)
plot.ylabel('Test Accuracy')
plot.title('Kernel Type')

plot.show()


## Now Let's Try SVM Regression 

predReg=[]

svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
svr_lin = SVR(kernel='linear', C=100, gamma='auto')
svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,
               coef0=1)

svr_rbf.fit(scaled_train,train_class)
for elem in test_cases:
   predReg.append(svr_rbf.predict([elem]))
print(predGau)


