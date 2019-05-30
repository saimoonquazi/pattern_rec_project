#!/usr/bin/env python
# coding: utf-8

# This script is used as a test bed to test the performance of the classifiers and feature reduction methods. Most of the functions used in this script belong to the sklearn library and any other function used has been developed by the project team. 

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
from sklearn.metrics import confusion_matrix  
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt


# Calling File reader, read the training data csv file and split train and validation set from the overall data

# In[2]:


filename='features_train.csv'
data,features,labels=data_file_reader.file_reader(filename,'train')
train_data_features, val_data_features, train_data_labels, val_data_labels = train_test_split(features, labels, test_size=0.4, random_state=10)


# Defining the classifiers to be used. Most of the parameters were kept as default and only slightly tuned to improve performance

# In[3]:


#Define the classifiers
svc=SVC(kernel='linear', C=1)
rf=RandomForestClassifier(n_estimators=50, random_state=1)
knn=KNeighborsClassifier(n_neighbors=3)
mv=VotingClassifier(estimators=[('rf', rf),('knn',knn),('svc',svc)], voting='hard')

#Define number of folds to be used for the cross validation. For this project, this section was ran twice, once with 40 folds, and then again with 10 folds
num_folds=10


# Performing cross validation on the classfiers to gauge performance

# In[4]:


#Conduct cross validation on the validation set and print out the accuracy and standard deviations for SVC Classifier (RAW)
cv_scores_svc=cross_val_score(svc, val_data_features,np.ravel(val_data_labels), cv=num_folds)
accur_crossval_svc=cv_scores_svc.mean()*100
std_crossval_svc=cv_scores_svc.std()*200
print('The Accuracy of the Support Vector Machine Classifier with Cross Validation is : %f'%accur_crossval_svc+'%'+' (+/- %0.2f)'%std_crossval_svc)

#Conduct cross validation on the validation set and print out the accuracy and standard deviations for Random Forest Classifier (RAW)
cv_scores_rf=cross_val_score(rf, val_data_features,np.ravel(val_data_labels), cv=num_folds)
accur_crossval_rf=cv_scores_rf.mean()*100
std_crossval_rf=cv_scores_rf.std()*200
print('The Accuracy of the Random Forest Classifier with Cross Validation is : %f'%accur_crossval_rf+'%'+' (+/- %0.2f)'%std_crossval_rf)

#Conduct cross validation on the validation set and print out the accuracy and standard deviations for KNN Classifier (RAW)
cv_scores_knn=cross_val_score(knn, val_data_features,np.ravel(val_data_labels), cv=num_folds)
accur_crossval_knn=cv_scores_knn.mean()*100
std_crossval_knn=cv_scores_knn.std()*200
print('The Accuracy of the K-Nearest Neighbour Classifier with Cross Validation is : %f'%accur_crossval_knn+'%'+' (+/- %0.2f)'%std_crossval_knn)

#Conduct cross validation on the validation set and print out the accuracy and standard deviations for Majority Voting Classifier (RAW)
cv_scores_mv=cross_val_score(mv, val_data_features,np.ravel(val_data_labels), cv=num_folds)
accur_crossval_mv=cv_scores_mv.mean()*100
std_crossval_mv=cv_scores_mv.std()*200
print('The Accuracy of the Majority Voting Classifier with Cross Validation is : %f'%accur_crossval_mv+'%'+' (+/- %0.2f)'%std_crossval_mv)


# Following procedure is to perform PCA on the data to perform feature reduction. 

# In[5]:


# Scale the input data (Good Practice when performing PCA)
sc=StandardScaler()
train_set=sc.fit_transform(val_data_features)

#Perform PCA on the input data reducing the input from 360 dimensions to 100 dimensions
pca=PCA(n_components=100)
pca_train_set= pca.fit_transform(train_set) 
print(pca.explained_variance_ratio_)  


# The following section conducts the cross validation once again, this time using the PCA transformed dataset to classify the data. The same number of folds as specified above. 

# In[6]:


#Conduct cross validation on the validation set and print out the accuracy and standard deviations for SVC Classifier (PCA)
cv_scores_svc_pca=cross_val_score(svc, pca_train_set,np.ravel(val_data_labels), cv=num_folds)
accur_crossval_svc_pca=cv_scores_svc_pca.mean()*100
std_crossval_svc_pca=cv_scores_svc_pca.std()*200
print('The Accuracy of the Support Vector Machine Classifier with Cross Validation is : %f'%accur_crossval_svc_pca+'%'+' (+/- %0.2f)'%std_crossval_svc_pca)

#Conduct cross validation on the validation set and print out the accuracy and standard deviations for SVC Classifier (PCA)
cv_scores_rf_pca=cross_val_score(rf, pca_train_set,np.ravel(val_data_labels), cv=num_folds)
accur_crossval_rf_pca=cv_scores_rf_pca.mean()*100
std_crossval_rf_pca=cv_scores_rf_pca.std()*200
print('The Accuracy of the Random Forest Classifier with Cross Validation is : %f'%accur_crossval_rf_pca+'%'+' (+/- %0.2f)'%std_crossval_rf_pca)

#Conduct cross validation on the validation set and print out the accuracy and standard deviations for SVC Classifier (PCA)
cv_scores_knn_pca=cross_val_score(knn, pca_train_set,np.ravel(val_data_labels), cv=num_folds)
accur_crossval_knn_pca=cv_scores_knn_pca.mean()*100
std_crossval_knn_pca=cv_scores_knn.std()*200
print('The Accuracy of the K-Nearest Neighbour Classifier with Cross Validation is : %f'%accur_crossval_knn_pca+'%'+' (+/- %0.2f)'%std_crossval_knn_pca)

#Conduct cross validation on the validation set and print out the accuracy and standard deviations for SVC Classifier (PCA)
cv_scores_mv_pca=cross_val_score(mv, pca_train_set,np.ravel(val_data_labels), cv=num_folds)
accur_crossval_mv_pca=cv_scores_mv_pca.mean()*100
std_crossval_mv_pca=cv_scores_mv_pca.std()*200
print('The Accuracy of the Majority Voting Classifier with Cross Validation is : %f'%accur_crossval_mv_pca+'%'+' (+/- %0.2f)'%std_crossval_mv_pca)


# Following procedure is to perform LDA on the data to perform feature reduction. 

# In[7]:


#Perform LDA transform on the input data to reduce the feature space from 360 to 200 dimensions
lda=LDA(n_components=200)
lda_train_set=lda.fit_transform(val_data_features,np.ravel(val_data_labels))


# The following section conducts the cross validation once again, this time using the LDA transformed dataset to classify the data. The same number of folds as specified above. 

# In[8]:


#Conduct cross validation on the validation set and print out the accuracy and standard deviations for SVC Classifier (LDA)
cv_scores_svc_lda=cross_val_score(svc, lda_train_set,np.ravel(val_data_labels), cv=num_folds)
accur_crossval_svc_lda=cv_scores_svc_lda.mean()*100
std_crossval_svc_lda=cv_scores_svc_lda.std()*200
print('The Accuracy of the Support Vector Machine Classifier +LDA with Cross Validation is : %f'%accur_crossval_svc_lda+'%'+' (+/- %0.2f)'%std_crossval_svc_lda)

#Conduct cross validation on the validation set and print out the accuracy and standard deviations for SVC Classifier (LDA)
cv_scores_rf_lda=cross_val_score(rf, lda_train_set,np.ravel(val_data_labels), cv=num_folds)
accur_crossval_rf_lda=cv_scores_rf_lda.mean()*100
std_crossval_rf_lda=cv_scores_rf_lda.std()*200
print('The Accuracy of the Random Forest Classifier + LDA with Cross Validation is : %f'%accur_crossval_rf_lda+'%'+' (+/- %0.2f)'%std_crossval_rf_lda)

#Conduct cross validation on the validation set and print out the accuracy and standard deviations for SVC Classifier (LDA)
cv_scores_knn_lda=cross_val_score(knn, lda_train_set,np.ravel(val_data_labels), cv=num_folds)
accur_crossval_knn_lda=cv_scores_knn_lda.mean()*100
std_crossval_knn_lda=cv_scores_knn_lda.std()*200
print('The Accuracy of the K-Nearest Neighbour Classifier + LDA with Cross Validation is : %f'%accur_crossval_knn_lda+'%'+' (+/- %0.2f)'%std_crossval_knn_lda)

#Conduct cross validation on the validation set and print out the accuracy and standard deviations for SVC Classifier (LDA)
cv_scores_mv_lda=cross_val_score(mv, lda_train_set,np.ravel(val_data_labels), cv=num_folds)
accur_crossval_mv_lda=cv_scores_mv_lda.mean()*100
std_crossval_mv_lda=cv_scores_mv_lda.std()*200
print('The Accuracy of the Majority Voting Classifier +LDA with Cross Validation is : %f'%accur_crossval_mv_lda+'%'+' (+/- %0.2f)'%std_crossval_mv_lda)


# Following procedure is to perform ICA on the data to perform feature reduction. 

# In[15]:


#Perform ICA to transfrom the dataset from 360 to 320 components
ica =  FastICA(n_components=320,random_state=0)
ica_train_set=ica.fit_transform(val_data_features,np.ravel(val_data_labels))


# The following section conducts the cross validation once again, this time using the ICA transformed dataset to classify the data. The same number of folds as specified above. 

# In[16]:


#Conduct cross validation on the validation set and print out the accuracy and standard deviations for SVC Classifier (ICA)
cv_scores_svc_ica=cross_val_score(svc, ica_train_set,np.ravel(val_data_labels), cv=num_folds)
accur_crossval_svc_ica=cv_scores_svc_ica.mean()*100
std_crossval_svc_ica=cv_scores_svc_ica.std()*200
print('The Accuracy of the Support Vector Machine Classifier + ICA with Cross Validation is : %f'%accur_crossval_svc_ica+'%'+' (+/- %0.2f)'%std_crossval_svc_ica)

#Conduct cross validation on the validation set and print out the accuracy and standard deviations for SVC Classifier (ICA)
cv_scores_rf_ica=cross_val_score(rf, ica_train_set,np.ravel(val_data_labels), cv=num_folds)
accur_crossval_rf_ica=cv_scores_rf_ica.mean()*100
std_crossval_rf_ica=cv_scores_rf_ica.std()*200
print('The Accuracy of the Random Forest Classifier + ICA with Cross Validation is : %f'%accur_crossval_rf_ica+'%'+' (+/- %0.2f)'%std_crossval_rf_ica)

#Conduct cross validation on the validation set and print out the accuracy and standard deviations for SVC Classifier (ICA)
cv_scores_knn_ica=cross_val_score(knn, ica_train_set,np.ravel(val_data_labels), cv=num_folds)
accur_crossval_knn_ica=cv_scores_knn_ica.mean()*100
std_crossval_knn_ica=cv_scores_knn_ica.std()*200
print('The Accuracy of the K-Nearest Neighbour Classifier + ICA with Cross Validation is : %f'%accur_crossval_knn_ica+'%'+' (+/- %0.2f)'%std_crossval_knn_ica)

#Conduct cross validation on the validation set and print out the accuracy and standard deviations for SVC Classifier (ICA)
cv_scores_mv_ica=cross_val_score(mv, ica_train_set,np.ravel(val_data_labels), cv=num_folds)
accur_crossval_mv_ica=cv_scores_mv_ica.mean()*100
std_crossval_mv_ica=cv_scores_mv_ica.std()*200
print('The Accuracy of the Majority Voting Classifier + ICA with Cross Validation is : %f'%accur_crossval_mv_ica+'%'+' (+/- %0.2f)'%std_crossval_mv_ica)


# The following simply prints out a table that makes the results easily readable and interpretable.

# In[17]:


classifiers=[' ', 'Support Vector Machine','Random Forest','K-Nearest Neighbour','Majority Voting']
feature_reducers=np.asarray(['No Feature Reduction','PCA','LDA','ICA'])
accuracy_values=[(classifiers),(feature_reducers[0],accur_crossval_svc,accur_crossval_rf,accur_crossval_knn,accur_crossval_mv),(feature_reducers[1],accur_crossval_svc_pca,accur_crossval_rf_pca,accur_crossval_knn_pca,accur_crossval_mv_pca),(feature_reducers[2],accur_crossval_svc_lda,accur_crossval_rf_lda,accur_crossval_knn_lda,accur_crossval_mv_lda),(feature_reducers[3],accur_crossval_svc_ica,accur_crossval_rf_ica,accur_crossval_knn_ica,accur_crossval_mv_ica)]
print(tabulate(accuracy_values))


# In[18]:


########################################################################################################################
# Function Name: plot_confusion_matrix
# Function Inputs : cm,classes,normalize,title, cmap
# Returns: ax
# Description: This function takes in a confusion matrix and plots it out in a matplot axis. This function is used as a 
#              helper to make a nice graphical representation of the confusion matrix. This function was obtained from 
#              https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
#########################################################################################################################
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = cm
    # Only use the labels that appear in the data
    classes = classes
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


# In[56]:


train_data_features, test_data_features, train_data_labels, test_data_labels = train_test_split(features, labels, test_size=0.1333333, random_state=7)


# In[57]:


lda=LDA(n_components=200)
lda_train_set_cm=lda.fit_transform(train_data_features,np.ravel(train_data_labels))
lda_test_set_cm = lda.transform(test_data_features)

#Define classes for Confusion matrix and evaluate and then print it out using the function above.
classes=['Natural Scene','Man Made Object in Natural Scene']
clf_cm=mv.fit(lda_train_set_cm,np.ravel(train_data_labels))
prediction_cm=clf_cm.predict(lda_test_set_cm)
cmat_mv=confusion_matrix(test_data_labels,prediction_cm)
plot_confusion_matrix(cmat_mv, classes=classes, title='Confusion matrix')
plt.savefig('MV confusion matrix.png')

#Evaluate performance of the Classifer with LDA
mv_lda_score=accuracy_score(test_data_labels,prediction_cm)
print('The Majority Voting Classifier Accuracy with LDA is: %f'%(mv_lda_score*100)+'%')


# In[58]:


#Compute the TP,TN,FP,FN from the confusion matrix
TP = cmat_mv[0,0]
TN = cmat_mv[1,1]
FP = cmat_mv[1,0]
FN = cmat_mv[0,1]

#Use these values to evalute the classifier's precision
Predicted_precision_mv_lda=precision_score(test_data_labels, prediction_cm, average='macro')
print('The Majority Voting Classifier with LDA Predicted Precision is: %f'%(Predicted_precision_mv_lda*100)+'%')

#Use these values to evalute the classifier's Sensitivity and Specificity
sensitivity=(TP/(TP+FN))
specificity=(TN/(TN+FP))
print('The Majority Voting Classifier with LDA Predicted Sensitivity is: %f'%(sensitivity*100)+'%')
print('The Majority Voting Classifier with LDA Predicted Specificity is: %f'%(specificity*100)+'%')


# In[ ]:




