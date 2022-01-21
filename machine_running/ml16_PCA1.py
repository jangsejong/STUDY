from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_boston
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.metrics import accuracy
from tensorflow.python.keras.utils.metrics_utils import result_wrapper

#1. 데이터
datasets= load_boston()
#print(datasets.DESCR)
x = datasets.data
y = datasets.target
#print(x.shape, y.shape)    #(178, 13) (178,)
#print(y) 0과 1로 수렴해서 셔플써야됨
#print(np.unique(y))  # [0, 1, 2]

#차원을 줄여준다.
from sklearn.decomposition import PCA
pca = PCA(n_components=8)
x = pca.fit_transform(x)

print(x.shape, y.shape)    #(506, 8) (506,)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=66)  #shuffle 은 기본값 True

# import 
print(sk.__version__)