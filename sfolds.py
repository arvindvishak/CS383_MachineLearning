#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import random
import math

# In[17]:


dataSet = np.genfromtxt('x06simple.csv', delimiter=',')

# In[18]:


dataSet = dataSet[1:, 1:]

# In[19]:


# set up the number of s-folds
s_folds = 20

# In[20]:


# seeding the random number generator
random.seed(0)

# In[21]:


np.random.shuffle(dataSet)

# In[22]:


# computing the length of each fold
fold_length = math.ceil(len(dataSet) / s_folds)

# In[23]:


print(fold_length)

# In[24]:


# initializing empty list to store the error values
errors_squared = []

# In[34]:


for i in range(s_folds):
    header = fold_length * (i - 1) + 1

    tail = min(header + fold_length - 1, len(dataSet))

    testing_data = dataSet[header:tail, :]

    training_data = np.vstack((dataSet[:header - 1, :], dataSet[tail:, :]))

    # now to standardize training and testing data (other than the last column)

    training_data_y = training_data[:, -1]

    training_data_y = np.reshape(training_data_y, (len(training_data_y), 1))

    training_data_x = training_data[:, :-1]
    testing_data_x = testing_data[:, :-1]

    testing_data_y = testing_data[:, -1]
    testing_data_y = np.reshape(testing_data_y, (len(testing_data_y), 1))

    average_training_x = np.mean(training_data_x, axis=0)
    std_training_x = np.std(training_data_x, axis=0)

    average_testing_x = np.mean(testing_data_x, axis=0)
    std_testing_x = np.std(testing_data_x, axis=0)

    x_training = (training_data_x - average_training_x) / std_training_x
    x_testing = (testing_data_x - average_training_x) / std_training_x

    training_ones = np.ones(len(x_training))
    training_ones = np.reshape(training_ones, (len(training_ones), 1))

    testing_ones = np.ones(len(x_testing))
    testing_ones = np.reshape(testing_ones, (len(testing_ones), 1))

    X = np.column_stack((training_ones, x_training))
    testX = np.column_stack((testing_ones, x_testing))


    xtx = np.dot(np.transpose(X), X)
    xtx_inv = np.linalg.inv(xtx)

    weights = np.dot(np.dot(xtx_inv, np.transpose(X)), training_data_y)

    predicted = np.matmul(testX, weights)
    print('jello')

# In[33]:


print('hell')
# In[ ]:




