import numpy as np
from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_wine
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.metrics import accuracy
from tensorflow.python.keras.utils.metrics_utils import result_wrapper

from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression #분류모델이다
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


#1. 데이터
datasets= load_wine()
#print(datasets.DESCR)
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split,KFold

n_splits =5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

model = SVC()

scores = cross_val_score(model, x, y, cv=kfold)
print("Acc :", scores)


