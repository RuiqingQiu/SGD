import numpy as np

n = 2
k = 2
learning_rate = 0.001
M = np.random.randn(n, n)
#M = np.random.rand(n,n)
print "M is ", M

v = np.random.rand(n,k)
u = np.random.rand(n,k)

for j in range(0, 10):
    for i in range(0, 100):
        u = u - learning_rate * ((M - u * np.transpose(v)) * v)
    for i in range(0, 100):
        v = v - learning_rate * ((M - u * np.transpose(v)) * u)
    #print u
    #print v
print u * np.transpose(v)
