#!/usr/bin/env python
# coding: utf-8

# In[35]:


import csv
import random
import numpy as np


# In[51]:

def file_reader():
    #Open the data file
    csv.register_dialect('myDialect',
    delimiter = '\t',
    skipinitialspace=True)

    #Read data and store the data
    data_table = []
    with open('features.csv', 'r') as csvFile:
        reader = csv.reader(csvFile, dialect='myDialect')
        data_table = list(reader)
    csvFile.close()

    data = np.zeros((len(data_table), len(data_table[0])))

    for i in range(0,len(data_table)):
        tmp = data_table[i]
        for j in range(0,len(tmp)):
            data[i,j] = float(tmp[j])
    np.random.shuffle(data)

    features=data[:,1:]
    labels=data[:,[0]]
    
    return data, features, labels
##