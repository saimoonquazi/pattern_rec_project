#!/usr/bin/env python
# coding: utf-8

#######################################################################################################################################
# This is a helper header script to streamline the reading of the data files into relevant groups (Features and labels). The functions # in this script makes it easier for the user to reuse the data reading functionality from the training or test file.  
#######################################################################################################################################

#Import Relevant Libraries
import csv
import random
import numpy as np

######################################################################################################################################
# Function Name: file_reader                                 
# Function Inputs : filename, train_test
# Returns: data, features, labels
# Description: This function takes in a filename to be read, along with a flag of whether the file represents test data or training    # dataset and return array containing the overall data in the file, only the features and the corresponding labels. The function also  # performs random shuffling of the data to ensure that regardless of whatever arrangement the data is represented in the CSV file, it   # return a randomized array of the features and labels. It should be noted that the random shuffling is skipped if the data is for test # as for this applcation, such files will contain a single sample. 
#######################################################################################################################################
def file_reader(filename,train_test):
    #Open the data file
    csv.register_dialect('myDialect',
    delimiter = '\t',
    skipinitialspace=True)

    #Read data and store the data
    data_table = []
    with open(filename, 'r') as csvFile:
        reader = csv.reader(csvFile, dialect='myDialect')
        data_table = list(reader)
    csvFile.close()

    #Define a container array for the data table
    data = np.zeros((len(data_table), len(data_table[0])))

    #Loop through the list of data extracted and assign them to corresponding positions in the array
    for i in range(0,len(data_table)):
        tmp = data_table[i]
        for j in range(0,len(tmp)):
            data[i,j] = float(tmp[j])
    
    #If the CSV file was for training, shuffle the array
    if(train_test=='train'):
        np.random.shuffle(data)

        #Seperate the features and labels
        features=data[:,1:]
        labels=data[:,[0]]
    #If the CSV file is for test, just seperate the features and labels    
    else:
        features=data
        labels=None
       
    return data, features, labels

