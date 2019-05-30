#!/usr/bin/env python
# coding: utf-8

# This script was used to loop through the images in respective directories of the training data and store the histogram angles to the feature_train.csv file which was then used to train the model.

# Import all relevant libraries

# In[3]:


import feature_extractor


# Specify the directories where the training images are kept and call the train_data_gen function from the feature extractor script with appropriate labels to be appended and the output file name to be used. Note: for overall training data this output filename should be the same. 

# In[4]:


man_made_obj_directory='Dataset/ManMadeScenesTrain/'
nature_directory='Dataset/NaturalScenesTrain/'

feature_extractor.train_data_gen(nature_directory,0,'features_train.csv')
feature_extractor.train_data_gen(man_made_obj_directory,1,'features_train.csv')


# In[ ]:




