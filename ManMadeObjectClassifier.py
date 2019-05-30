#!/usr/bin/env python
# coding: utf-8

# This is a test script written to manually test out the classification with a given test image. The script contains a training block and a test block with the relevant feature reduction. This is a simplified version of the methodology followed for the testing phase.

# Import all relevant libraries required

# In[1]:


import csv
import random
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import data_file_reader
import matplotlib.pyplot as plt
import feature_extractor
import os


# **------------------------------------------Training Block -----------------------------------------------------------**

# Read the training data from the Training CSV File & Seperate Features and labels for training

# In[2]:


filename='features_train.csv'
data,features,labels=data_file_reader.file_reader(filename,'train')


# Define the classifiers used

# In[3]:


svc=SVC(kernel='linear', C=1)
rf=RandomForestClassifier(n_estimators=50, random_state=1)
knn=KNeighborsClassifier(n_neighbors=3)
mv=VotingClassifier(estimators=[('rf', rf),('knn',knn),('svc',svc)], voting='hard')


# LDA Transform the data and fit the model with it

# In[7]:


lda=LDA(n_components=200)
lda_train_set=lda.fit_transform(features,np.ravel(labels))
clf=mv.fit(lda_train_set,np.ravel(labels))


# **----------------------------------------Prediction Block------------------------------------------------------------**

# Check if the feature_test csv file exists, if so, delete the current file and run the feature extractor algorithm on the given filename. The feature extractor function will replace a file with the same name, but the prediction step should only have 1 set of features to make the prediction, to ensure that the right file is being used for predictions.

# In[5]:


filename='features_test.csv'
exists = os.path.isfile(filename)
if exists:
    os.remove(filename) 
#Execute the feature extraction function from the appropriate header file
feature_extractor.extract_features_prediction('IMG_20181022_225217_392.jpg')
#Read the features_test csv file and store the feature space as predict_features 
data,predict_features,_=data_file_reader.file_reader(filename,'test')


# Make Prediction based on the Transformed Data

# In[8]:


lda_test_set = lda.transform(predict_features)
prediction=clf.predict(lda_test_set)
if prediction==0:
    print('The Image Contains a Natural Scene')
else:
    print('The Image Contains Man Made Objects in the Scene')


# In[ ]:




