import numpy as np
from matplotlib import pyplot as plt

def entropy(x):
    return x*np.log2(x)

seq = range(5,95, 5)
lst=[]
xlist =[]
for item in seq:
    item /= 100
    xlist.append(item)
    lst.append(entropy(item))

# print("lst:", lst)
# print("xlist", xlist)
# plt.plot(xlist, lst)
# plt.show()

entropy(1)
entropy(0.1)

tuple(lst)

np.random.randn(3, 2)

from sklearn import metrics
y = np.array([1, 1, 2, 2])
scores = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)


import numpy as np 
from sklearn import metrics 
y = np.array([1, 1, 2, 2]) 
scores = np.array([0.1, 0.4, 0.35, 0.8]) 
fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2) 
print("fpr: ", fpr)
print("tpr:",tpr)
plt.plot(fpr, tpr)
plt.show()

lst = [0,1,2,3,4,5,6,7]
lst[0:2]
lst[3:6]=(30,40,50)
lst
import numpy as np
arr = np.array[[1,2,3],[4,5,6]]
arr.shape()
arr1 = np.ones((2,))
arr1

arr2 = np.ones((1,2))
arr2

arr3 = np.ones((2, 3))
arr3

arr1.dot(arr3)

arr1*arr2

import numpy as np
arr3 = [[1,2,3],[4,5,6]]
np.zeros_like(arr3)