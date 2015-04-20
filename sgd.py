import math
import numpy as np
import re
import os.path

# test result function
def frobenius_norm(M1, M2):
    # print "M1 is"
    # print M1
    # print "M2 is"
    # print M2
    total = 0.0
    for a,b in zip(M1, M2):
        for c, d in zip(a, b):
            total += (c - d)*(c - d)
            # print "total, ", total
    return total

# dimension of each example in the dataset
d = 15
# k is the top pricipal calculated by the eigenvectors, right now just fix it
k = 5
# fix the learning rate
#learning_rate = 0.00000000001
learning_rate = 0.0000000000001

print "lr, ", learning_rate
# preprocess the data
data_set = []
data_set_done = []
if os.path.exists('new_data'):
    with open('new_data') as data:
        for line in data:
            data_set_done.append(map(float, line.split()))
            #data_set_done.append([map(float, line.split())[0], map(float, line.split())[1], map(float, line.split())[2], map(float, line.split())[3]])
    #print data_set_done
else:
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


    print "data before processed dimension is ", len(modified_data_set[0])
    #print modified_data_set
    item_list = []
    count = 0
    dimension_to_increase = 0
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
        dimension_to_increase = dimension_to_increase + len(items)-1
        #print "item is "
        #print items
        #print "\n\n"
        # loop through the whole data set to replace the feature
        for j in range(len(modified_data_set)):
            v = modified_data_set[j]
            #print v
            # get the item in the vector
            #print "accessing ", i+count
            tmp = v[i+count]
            index  = items.index(tmp)
            front = v[0:i+count]
            end = v[i+1+count:len(v)]
            zeros = [0]*len(items)
            zeros[index] = 1
            front.extend(zeros)
            front.extend(end)
            modified_data_set[j] = front
            #print "result is ", modified_data_set[j]
            #v[i] = float(index)
        count = count + len(items)-1
    for v in modified_data_set:
        v = map(float, v)
        data_set_done.append(v)
    print "dimension to increase ", dimension_to_increase
    print "data processed dimension is ", len(data_set_done[0])
    #print data_set_done
    print "size of data set done is ", len(data_set_done)

f = open('new_data', 'w')
for vector in data_set_done:
    for num in vector:
        f.write(str(num)+" ")
    f.write("\n")
f.close()

print data_set_done[0]
# U and V are d x k dimension matrix
v = np.random.rand(len(data_set_done[0]),k)

print "V start with, ", v
u = np.random.rand(len(data_set_done[0]),k)
print "U start with, ", u

# # data set size
N = len(data_set_done)
distance = []

# M = np.array(data_set_done[0]) * np.transpose(np.array(data_set_done[0])) / N
# for i in range(1, len(data_set_done)):
#     M = M + (np.array([data_set_done[i]]) * np.transpose(np.array([data_set_done[i]]))) / N
# print "Covariance Matrix is: ", M

# # U, s, V = np.linalg.svd(M, full_matrices=True)

# # distance.append(frobenius_norm(M, np.dot(U,np.transpose(V))))
# print distance

# #SGD function
# #run the whole optimization process 10 times
# converged = False
# last_distance = 0.0
# while not converged:
#     for t in range(0, 100):
#         u = u + learning_rate * np.dot(M - np.dot(u,np.transpose(v)), v)
#     for t in range(0, 100):
#         v = v + learning_rate * np.dot(M - np.dot(u, np.transpose(v)), u)
#     result = frobenius_norm(M, np.dot(u,np.transpose(v)))
#     print result
#     if abs(result-last_distance) < 0.001:
#         print "converged"
#         converged = True
#     last_distance = result
#     distance.append(result)
# print distance

#Stochastic gradient descent(SGD)
#Apply random permutation to the data
# permutated_data = np.random.permutation(data_set_done)
# print permutated_data[0]
# index = 0
# #print permutated_data
# converged = False
# last_distance = 0.0
# for i in range(0, 100):
#     index = (index + 1) % N
#     #M = np.array(permutated_data[index]) * np.transpose(np.array(permutated_data[index]))
#     M = np.array([permutated_data[index]]) * np.transpose(np.array([permutated_data[index]]))
#     for t in range(0, 100):
#         u = u + learning_rate * np.dot(M - np.dot(u,np.transpose(v)), v)
#     for t in range(0, 100):
#         v = v + learning_rate * np.dot(M - np.dot(u, np.transpose(v)), u)
#     #print np.dot(u, np.transpose(v))
#     result = frobenius_norm(M, np.dot(u,np.transpose(v)))
#     print result
#     if abs(result-last_distance) < 0.001:
#         print "converged"
#         converged = True
#     last_distance = result
#     distance.append(result)
# print distance

def length(data):
    sum = 0.0
    for i in data:
        sum += i * i
    return math.sqrt(sum)

# Normalize all data
for data in data_set_done:
    print length(data)

d = len(data_set_done[0]) #shape
theta = 2
s = np.random.gamma(d, 2, d)

print s
