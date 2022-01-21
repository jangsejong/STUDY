from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import datasets
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.metrics import accuracy
from tensorflow.python.keras.utils.metrics_utils import result_wrapper

#1. 데이터

(x_train, _ ), (x_test, _ ) = mnist.load_data()

x = np.append(x_train, x_test, axis=0)

print(x_train.shape, x_test.shape)    #(60000, 28, 28) (10000, 28, 28)

x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])


from sklearn.decomposition import PCA
pca = PCA(n_components=784)
x = pca.fit_transform(x)

pca_EVR = pca.explained_variance_ratio_

cumsum = np.cumsum(pca_EVR)

# print(cumsum)

print(np.argmax(cumsum >=0.95)+1)  #154
print(np.argmax(cumsum >=0.99)+1)  #331
print(np.argmax(cumsum >=0.999)+1)  #486
print(np.argmax(cumsum)+1)  #713   #  1.0







# x = np.argmax(x , axis=1)

# x_train = x_train.reshape(60000, 784)

# #차원을 줄여준다.
# from sklearn.decomposition import PCA
# pca = PCA(n_components=28)
# x = pca.fit_transform(x)

# # #
# pca_EVR = pca.explained_variance_ratio_

# print(pca_EVR)

# cumsum = np.cumsum(pca_EVR)

# # print(cumsum)

# import matplotlib.pyplot as plt

# plt.plot(cumsum)
# plt.grid()
# plt.show()




# '''
# 0.95 n_components > 17
# 0.99  n_components > 21
# 1.0 n_components 
# '''



# # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=66)  #shuffle 은 기본값 True
