import numpy as np
from sklearn.utils import shuffle

l1 = np.zeros(10)
l2 = np.ones(10)

print(l1, l2, end="\n")

l3 = np.append(l1, l2)
print(l3)