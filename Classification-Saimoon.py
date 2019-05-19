#!/usr/bin/env python
# coding: utf-8

# Importing Relevant Libraries

# In[ ]:


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


# Calling File reader and splitting train and test set from the overall data

# In[9]:


data,features,labels=data_file_reader.file_reader()
train_data_features, test_data_features, train_data_labels, test_data_labels = train_test_split(features, labels, test_size=0.2, random_state=10)


# Defining the classifiers to be used

# In[10]:


svc=SVC(kernel='linear', C=1)
rf=RandomForestClassifier(n_estimators=50, random_state=1)
knn=KNeighborsClassifier(n_neighbors=3)
mv=VotingClassifier(estimators=[('rf', rf),('knn',knn),('svc',svc)], voting='hard')


# Performing cross validation on the classfiers to gauge performance

# In[12]:


cv_scores_svc=cross_val_score(svc, train_data_features,np.ravel(train_data_labels), cv=10)
accur_crossval_svc=cv_scores_svc.mean()*100
std_crossval_svc=cv_scores_svc.std()*2
print('The Accuracy of the Support Vector Machine Classifier with 10-fold Cross Validation is : %f'%accur_crossval_svc+'%'+' (+/- %0.2f)'%std_crossval_svc)

cv_scores_rf=cross_val_score(rf, train_data_features,np.ravel(train_data_labels), cv=10)
accur_crossval_rf=cv_scores_rf.mean()*100
std_crossval_rf=cv_scores_rf.std()*2
print('The Accuracy of the Random Forest Classifier with 10-fold Cross Validation is : %f'%accur_crossval_rf+'%'+' (+/- %0.2f)'%std_crossval_rf)

cv_scores_knn=cross_val_score(knn, train_data_features,np.ravel(train_data_labels), cv=10)
accur_crossval_knn=cv_scores_knn.mean()*100
std_crossval_knn=cv_scores_knn.std()*2
print('The Accuracy of the K-Nearest Neighbour Classifier with 10-fold Cross Validation is : %f'%accur_crossval_knn+'%'+' (+/- %0.2f)'%std_crossval_knn)

cv_scores_mv=cross_val_score(mv, train_data_features,np.ravel(train_data_labels), cv=10)
accur_crossval_mv=cv_scores_mv.mean()*100
std_crossval_mv=cv_scores_mv.std()*2
print('The Accuracy of the Majority Voting Classifier with 10-fold Cross Validation is : %f'%accur_crossval_mv+'%'+' (+/- %0.2f)'%std_crossval_mv)


# Following are two procedures to perform PCA and LDA on the data to perform feature reduction. 

# In[ ]:


# Scale the input data (Good Practice when performing PCA)
sc=StandardScaler()
train_set=sc.fit_transform(train_data_features)
test_set=sc.fit_transform(test_data_features)

#Perform PCA on the input data reducing the input from 4 dimensions to 2 dimensions
pca=PCA(n_components=40)
pca_train_set= pca.fit_transform(train_set) 
pca_test_set=pca.fit_transform(test_set)
print(pca.explained_variance_ratio_)  


# In[ ]:


lda=LDA(n_components=20)
lda_train_set=lda.fit_transform(train_data_features,np.ravel(train_data_labels))
lda_test_set = lda.transform(test_data_features)

