#!/usr/bin/env python
# coding: utf-8

# Importing Relevant Libraries

# In[2]:


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

from sklearn.decomposition import FastICA
from sklearn.datasets import load_digits


# Calling File reader and splitting train and test set from the overall data

# In[3]:


filename='features_train.csv'
data,features,labels=data_file_reader.file_reader(filename)
train_data_features, test_data_features, train_data_labels, test_data_labels = train_test_split(features, labels, test_size=0.2, random_state=10)


# Defining the classifiers to be used

# In[4]:


svc=SVC(kernel='linear', C=1)
rf=RandomForestClassifier(n_estimators=50, random_state=1)
knn=KNeighborsClassifier(n_neighbors=3)
mv=VotingClassifier(estimators=[('rf', rf),('knn',knn),('svc',svc)], voting='hard')
print(mv)

# Performing cross validation on the classfiers to gauge performance

# In[5]:


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


# Test Accuracy

# In[12]:


rf_clf=rf.fit(train_data_features,np.ravel(train_data_labels))
prediction_rf=rf_clf.predict(test_data_features)
rf_score=accuracy_score(test_data_labels,prediction_rf)
print('The Random Forest Classifier Accuracy is: %f'%(rf_score*100)+'%')


# Following are two procedures to perform PCA and LDA on the data to perform feature reduction. 

# In[6]:


# Scale the input data (Good Practice when performing PCA)
sc=StandardScaler()
train_set=sc.fit_transform(train_data_features)
test_set=sc.fit_transform(test_data_features)

#Perform PCA on the input data reducing the input from 4 dimensions to 2 dimensions
pca=PCA(n_components=80)
pca_train_set= pca.fit_transform(train_set) 
pca_test_set=pca.fit_transform(test_set)
print(pca.explained_variance_ratio_)


# In[16]:


rf_pca=rf.fit(pca_train_set,np.ravel(train_data_labels))
prediction_pca=rf_pca.predict(pca_test_set)
rf_pca_score=accuracy_score(test_data_labels,prediction_pca)
print('The Random Forest Classifier Accuracy with PCA is: %f'%(rf_pca_score*100)+'%')


# In[17]:


lda=LDA(n_components=200)
lda_train_set=lda.fit_transform(train_data_features,np.ravel(train_data_labels))
lda_test_set = lda.transform(test_data_features)

rf_lda=rf.fit(lda_train_set,np.ravel(train_data_labels))
prediction_lda=rf_lda.predict(lda_test_set)
rf_lda_score=accuracy_score(test_data_labels,prediction_lda)
print('The Random Forest Classifier Accuracy with LDA is: %f'%(rf_lda_score*100)+'%')


# In[ ]:

# FastICA

ica =  FastICA(n_components=320,random_state=0)
ica_train_set=ica.fit_transform(train_data_features,np.ravel(train_data_labels))
ica_test_set = ica.transform(test_data_features)

svc_ica=svc.fit(ica_train_set,np.ravel(train_data_labels))
prediction_svc_ica=svc_ica.predict(ica_test_set)
svc_ica_score=accuracy_score(test_data_labels,prediction_svc_ica)
print('The Support Vector Machine Classifier Accuracy with FastICA is: %f'%(svc_ica_score*100)+'%')

cv_scores_svc_ica=cross_val_score(svc, ica_train_set,np.ravel(train_data_labels), cv=10)
accur_crossval_svc_ica=cv_scores_svc_ica.mean()*100
std_crossval_svc_ica=cv_scores_svc_ica.std()*2
print('The Accuracy of the Support Vector Machine Classifier + ICA with 10-fold Cross Validation is : %f'%accur_crossval_svc_ica+'%'+' (+/- %0.2f)'%std_crossval_svc_ica)

rf_ica=rf.fit(ica_train_set,np.ravel(train_data_labels))
prediction_rf_ica=rf_ica.predict(ica_test_set)
rf_ica_score=accuracy_score(test_data_labels,prediction_rf_ica)
print('The Random Forest Classifier Accuracy with FastICA is: %f'%(rf_ica_score*100)+'%')

cv_scores_rf_ica=cross_val_score(rf, ica_train_set,np.ravel(train_data_labels), cv=10)
accur_crossval_rf_ica=cv_scores_rf_ica.mean()*100
std_crossval_rf_ica=cv_scores_rf_ica.std()*2
print('The Accuracy of the Random Forest Classifier + ICA with 10-fold Cross Validation is : %f'%accur_crossval_rf_ica+'%'+' (+/- %0.2f)'%std_crossval_rf_ica)

knn_ica=knn.fit(ica_train_set,np.ravel(train_data_labels))
prediction_knn_ica=knn_ica.predict(ica_test_set)
knn_ica_score=accuracy_score(test_data_labels,prediction_knn_ica)
print('The K-Nearest Neighbour Classifier Accuracy with FastICA is: %f'%(knn_ica_score*100)+'%')

cv_scores_knn_ica=cross_val_score(knn, train_data_features,np.ravel(train_data_labels), cv=20)
accur_crossval_knn_ica=cv_scores_knn_ica.mean()*100
std_crossval_knn_ica=cv_scores_knn_ica.std()*200
print('The Accuracy of the K-Nearest Neighbour Classifier + ICA with 10-fold Cross Validation is : %f'%accur_crossval_knn_ica+'%'+' (+/- %0.2f)'%std_crossval_knn_ica)


