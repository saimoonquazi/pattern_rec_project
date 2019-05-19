import data_file_reader as fread
import numpy as np

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def randomForestClas(train_features, train_labels, test_features, test_labels):
    random_forest = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
    random_forest.fit(train_features, train_labels.ravel())
    scores = cross_val_score(random_forest,test_features, train_labels.ravel(), cv=10)
    print(" Accuracy = ", scores.sum()/len(scores))
    return 1
##

def lda(train_features,train_labels, test_features):
    lda = LDA()
    
    lda_train_set = lda.fit(train_features,np.ravel(train_labels))
    lda_test_set = lda.transform(test_features)
    
    return lda_train_set, lda_test_set
##

def x10crossValidation(data):
    for i in range(1,10):
        data_split = i/10
        samples_higher_thr = int(len(data) * i+1/10)
        samples_lower_thr = int(len(data) * (i)/10)
        
        if i != 0:
            train_data = np.concatenate((np.array(data[0:samples_lower_thr,:]), data[samples_higher_thr+1:len(data),:]), axis=0)
        else:
            train_data = data[samples_higher_thr+1:len(data),:]
        ##
        
        test_data = data[samples_lower_thr:samples_higher_thr, :]
        
        train_features = train_data[:,1:]
        train_labels = train_data[:,[0]]
        test_features = test_data[:,1:]
        test_labels = test_data[:,[0]]
        
        lda_train_set, lda_test_set = lda(train_features, train_labels, test_features)
        
        #randomForestClas(train_features, train_labels, test_features, test_labels)
    ##
    return 1
##

def main():
    data, features, labels = fread.file_reader()
    
    #lda_features = lda(features, labels)
    #randomForestClas(features, labels)
    
    x10crossValidation(data)
    
    return 1
##

main()