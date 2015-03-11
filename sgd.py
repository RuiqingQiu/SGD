import math
import numpy as np
import re

# test result function
def frobenius_norm(M1, M2):
    total = 0.0
    for a,b in zip(M1, M2):
        for c, d in zip(a, b):
            total += (c - d)*(c - d)
    return total

# dimension of each example in the dataset
d = 15
# k is the top pricipal calculated by the eigenvectors, right now just fix it
k = 5
# fix the learning rate 
learning_rate = 0.001
# preprocess the data
data_set = []
with open('adult.data.txt') as data:
    for line in data:
        # '\s' matches whitespace
        tmp = re.sub(r'\s', '', line).split(',')
        data_set.append(tmp)
#print data_set

feature_vector_size = len(data_set[0])
print "feature vector is  " ,feature_vector_size
print "data set size is ", len(data_set)
tmp = data_set[0]

# find out which feature is not a digit
non_digit_index = []
for i in range (0,feature_vector_size):
    if tmp[i].isdigit():
        continue
    else:
        non_digit_index.append(i)
print non_digit_index

# make each string feature to become a number

# Copy
modified_data_set= []
for item in data_set:
    modified_data_set.append(item)
    
#print modified_data_set
# for all the non_digit index, count how many differnt items
for i in non_digit_index:
    # clear the item list to find different string for same feature
    items = []
    # loop through the whole data set
    for vector in data_set:
        # get the corresponsding item
        tmp = vector[i]
        # it already in the list
        if tmp in items:
            continue
        else:
            # otherwise append to the list
            items.append(tmp)
    print items
    # loop through the whole data set to replace the feature
    for v in modified_data_set:
        # get the item in the vector
        tmp = v[i]
        index  = items.index(tmp)
        v[i] = float(index)
data_set_done = []
for v in modified_data_set:
    v = map(float, v)
    data_set_done.append(v)
#print data_set_done
print "size of data set done is ", len(data_set_done)

# U and V are d x k dimension matrix
v = np.random.rand(d,k)
print "V start with, ", v
u = np.random.rand(d,k)
print "U start with, ", u

print "result of the caulcation is ", np.transpose(data_set_done[0]) 

# SGD function
# run the whole optimization process 10 times
for j in range(0, 10):
    # do u 100 rounds 
    for counter in range (0,99):
        # update u len(data_set) iterations
        for t in range(0, len(data_set_done)):
            #print (data_set[t] * np.transpose(data_set[t]) - np.dot(u,np.transpose(v)))
            # need to do np.dot() for matrix multiplication
            u = u - learning_rate * np.dot((data_set_done[t] * np.transpose(data_set_done[t]) - np.dot(u,np.transpose(v))), v)
    # do v 100 rounds
    for counter in range (0,99):
        # update v len(data_set) iterations
        for t in range(0, len(data_set_done)):
            v = v - learning_rate * np.dot((data_set_done[t] * np.transpose(data_set_done[t]) - np.dot(u, np.transpose(v))), u)
    #print frobenius_norm(M, np.dot(u,np.transpose(v)))


    
#print u * np.transpose(v)
