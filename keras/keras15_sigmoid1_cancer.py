import numpy as np
from sklearn.datasets import load_breast_cancer
datasets = load_breast_cancer
#print(datasets)

x = datasets.data
y = datasets.target
# print(x.shape, y.shape)     (569, 30),(569, )

#print(y) 이진분류, sigmoid
print(np.unique(y))   #[0, 1]