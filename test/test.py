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

plt.plot(xlist, lst)
plt.show()

entropy(1)
entropy(0.1)

tuple(lst)



