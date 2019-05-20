#!/usr/bin/env python
# coding: utf-8

# Importing Relevant Libraries

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
from sklearn.decomposition import FastICA
from sklearn.datasets import load_digits
from tabulate import tabulate


# Calling File reader and splitting train and test set from the overall data

# In[2]:


filename='features_train.csv'
data,features,labels=data_file_reader.file_reader(filename,'train')
#train_data_features, test_data_features, train_data_labels, test_data_labels = train_test_split(features, labels, test_size=0.2, random_state=10)


# Defining the classifiers to be used

# In[3]:


svc=SVC(kernel='linear', C=1)
rf=RandomForestClassifier(n_estimators=50, random_state=1)
knn=KNeighborsClassifier(n_neighbors=3)
mv=VotingClassifier(estimators=[('rf', rf),('knn',knn),('svc',svc)], voting='hard')
num_folds=10


# Performing cross validation on the classfiers to gauge performance

# In[22]:



cv_scores_svc=cross_val_score(svc, features,np.ravel(labels), cv=num_folds)
accur_crossval_svc=cv_scores_svc.mean()*100
std_crossval_svc=cv_scores_svc.std()*200
print('The Accuracy of the Support Vector Machine Classifier with 10-fold Cross Validation is : %f'%accur_crossval_svc+'%'+' (+/- %0.2f)'%std_crossval_svc)

cv_scores_rf=cross_val_score(rf, features,np.ravel(labels), cv=num_folds)
accur_crossval_rf=cv_scores_rf.mean()*100
std_crossval_rf=cv_scores_rf.std()*200
print('The Accuracy of the Random Forest Classifier with 10-fold Cross Validation is : %f'%accur_crossval_rf+'%'+' (+/- %0.2f)'%std_crossval_rf)

cv_scores_knn=cross_val_score(knn, features,np.ravel(labels), cv=num_folds)
accur_crossval_knn=cv_scores_knn.mean()*100
std_crossval_knn=cv_scores_knn.std()*200
print('The Accuracy of the K-Nearest Neighbour Classifier with 10-fold Cross Validation is : %f'%accur_crossval_knn+'%'+' (+/- %0.2f)'%std_crossval_knn)

cv_scores_mv=cross_val_score(mv, features,np.ravel(labels), cv=num_folds)
accur_crossval_mv=cv_scores_mv.mean()*100
std_crossval_mv=cv_scores_mv.std()*200
print('The Accuracy of the Majority Voting Classifier with 10-fold Cross Validation is : %f'%accur_crossval_mv+'%'+' (+/- %0.2f)'%std_crossval_mv)


# Following are two procedures to perform PCA and LDA on the data to perform feature reduction. 

# In[23]:


# Scale the input data (Good Practice when performing PCA)
sc=StandardScaler()
train_set=sc.fit_transform(features)
#test_set=sc.fit_transform(test_data_features)

#Perform PCA on the input data reducing the input from 4 dimensions to 2 dimensions
pca=PCA(n_components=80)
pca_train_set= pca.fit_transform(train_set) 
#pca_test_set=pca.fit_transform(test_set)
print(pca.explained_variance_ratio_)  


# In[24]:


cv_scores_svc_pca=cross_val_score(svc, pca_train_set,np.ravel(labels), cv=num_folds)
accur_crossval_svc_pca=cv_scores_svc_pca.mean()*100
std_crossval_svc_pca=cv_scores_svc_pca.std()*200
print('The Accuracy of the Support Vector Machine Classifier with 10-fold Cross Validation is : %f'%accur_crossval_svc_pca+'%'+' (+/- %0.2f)'%std_crossval_svc_pca)

cv_scores_rf_pca=cross_val_score(rf, pca_train_set,np.ravel(labels), cv=num_folds)
accur_crossval_rf_pca=cv_scores_rf_pca.mean()*100
std_crossval_rf_pca=cv_scores_rf_pca.std()*200
print('The Accuracy of the Random Forest Classifier with 10-fold Cross Validation is : %f'%accur_crossval_rf_pca+'%'+' (+/- %0.2f)'%std_crossval_rf_pca)

cv_scores_knn_pca=cross_val_score(knn, pca_train_set,np.ravel(labels), cv=num_folds)
accur_crossval_knn_pca=cv_scores_knn_pca.mean()*100
std_crossval_knn_pca=cv_scores_knn.std()*200
print('The Accuracy of the K-Nearest Neighbour Classifier with 10-fold Cross Validation is : %f'%accur_crossval_knn_pca+'%'+' (+/- %0.2f)'%std_crossval_knn_pca)

cv_scores_mv_pca=cross_val_score(mv, pca_train_set,np.ravel(labels), cv=num_folds)
accur_crossval_mv_pca=cv_scores_mv_pca.mean()*100
std_crossval_mv_pca=cv_scores_mv_pca.std()*200
print('The Accuracy of the Majority Voting Classifier with 10-fold Cross Validation is : %f'%accur_crossval_mv_pca+'%'+' (+/- %0.2f)'%std_crossval_mv_pca)


# In[4]:


lda=LDA(n_components=200)
lda_train_set=lda.fit_transform(features,np.ravel(labels))
#lda_test_set = lda.transform(test_data_features)

cv_scores_svc_lda=cross_val_score(svc, lda_train_set,np.ravel(labels), cv=num_folds)
accur_crossval_svc_lda=cv_scores_svc_lda.mean()*100
std_crossval_svc_lda=cv_scores_svc_lda.std()*2
print('The Accuracy of the Support Vector Machine Classifier +LDA with 10-fold Cross Validation is : %f'%accur_crossval_svc_lda+'%'+' (+/- %0.2f)'%std_crossval_svc_lda)

cv_scores_rf_lda=cross_val_score(rf, lda_train_set,np.ravel(labels), cv=num_folds)
accur_crossval_rf_lda=cv_scores_rf_lda.mean()*100
std_crossval_rf_lda=cv_scores_rf_lda.std()*2
print('The Accuracy of the Random Forest Classifier + LDA with 10-fold Cross Validation is : %f'%accur_crossval_rf_lda+'%'+' (+/- %0.2f)'%std_crossval_rf_lda)

cv_scores_knn_lda=cross_val_score(knn, lda_train_set,np.ravel(labels), cv=num_folds)
accur_crossval_knn_lda=cv_scores_knn_lda.mean()*100
std_crossval_knn_lda=cv_scores_knn_lda.std()*2
print('The Accuracy of the K-Nearest Neighbour Classifier + LDA with 10-fold Cross Validation is : %f'%accur_crossval_knn_lda+'%'+' (+/- %0.2f)'%std_crossval_knn_lda)

cv_scores_mv_lda=cross_val_score(mv, lda_train_set,np.ravel(labels), cv=num_folds)
accur_crossval_mv_lda=cv_scores_mv_lda.mean()*100
std_crossval_mv_lda=cv_scores_mv_lda.std()*2
print('The Accuracy of the Majority Voting Classifier +LDA with 10-fold Cross Validation is : %f'%accur_crossval_mv_lda+'%'+' (+/- %0.2f)'%std_crossval_mv_lda)


# In[26]:


ica =  FastICA(n_components=320,random_state=0)
ica_train_set=ica.fit_transform(features,np.ravel(labels))
#ica_test_set = ica.transform(test_data_features)

cv_scores_svc_ica=cross_val_score(svc, ica_train_set,np.ravel(labels), cv=num_folds)
accur_crossval_svc_ica=cv_scores_svc_ica.mean()*100
std_crossval_svc_ica=cv_scores_svc_ica.std()*200
print('The Accuracy of the Support Vector Machine Classifier + ICA with 10-fold Cross Validation is : %f'%accur_crossval_svc_ica+'%'+' (+/- %0.2f)'%std_crossval_svc_ica)

cv_scores_rf_ica=cross_val_score(rf, ica_train_set,np.ravel(labels), cv=num_folds)
accur_crossval_rf_ica=cv_scores_rf_ica.mean()*100
std_crossval_rf_ica=cv_scores_rf_ica.std()*200
print('The Accuracy of the Random Forest Classifier + ICA with 10-fold Cross Validation is : %f'%accur_crossval_rf_ica+'%'+' (+/- %0.2f)'%std_crossval_rf_ica)

cv_scores_knn_ica=cross_val_score(knn, ica_train_set,np.ravel(labels), cv=num_folds)
accur_crossval_knn_ica=cv_scores_knn_ica.mean()*100
std_crossval_knn_ica=cv_scores_knn_ica.std()*200
print('The Accuracy of the K-Nearest Neighbour Classifier + ICA with 10-fold Cross Validation is : %f'%accur_crossval_knn_ica+'%'+' (+/- %0.2f)'%std_crossval_knn_ica)

cv_scores_mv_ica=cross_val_score(mv, ica_train_set,np.ravel(labels), cv=num_folds)
accur_crossval_mv_ica=cv_scores_mv_ica.mean()*100
std_crossval_mv_ica=cv_scores_mv_ica.std()*200
print('The Accuracy of the Majority Voting Classifier + ICA with 10-fold Cross Validation is : %f'%accur_crossval_mv_ica+'%'+' (+/- %0.2f)'%std_crossval_mv_ica)


# In[28]:


classifiers=[' ', 'Support Vector Machine','Random Forest','K-Nearest Neighbour','Majority Voting']
feature_reducers=np.asarray(['No Feature Reduction','PCA','LDA','ICA'])
accuracy_values=[(classifiers),(feature_reducers[0],accur_crossval_svc,accur_crossval_rf,accur_crossval_knn,accur_crossval_mv),(feature_reducers[1],accur_crossval_svc_pca,accur_crossval_rf_pca,accur_crossval_knn_pca,accur_crossval_mv_pca),(feature_reducers[2],accur_crossval_svc_lda,accur_crossval_rf_lda,accur_crossval_knn_lda,accur_crossval_mv_lda),(feature_reducers[3],accur_crossval_svc_ica,accur_crossval_rf_ica,accur_crossval_knn_ica,accur_crossval_mv_ica)]
print(tabulate(accuracy_values))
#print(accuracy_values)


# In[9]:


import feature_extractor
feature_extractor.extract_features_prediction('cps201410067335.ppm')
filename='features_test.csv'
data,predict_features,_=data_file_reader.file_reader(filename,'test')


# In[10]:


lda_test_set = lda.transform(predict_features)
clf=mv.fit(lda_train_set,np.ravel(labels))
prediction=clf.predict(lda_test_set)
print(prediction)


# In[ ]:




