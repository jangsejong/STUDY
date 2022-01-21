# n_component > 0.95 이상   '154'
# xgboost, gridSearch 또는 RandomSearch를 쓸 것
# m16결과를 뛰어넘어라

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn import datasets
from tensorflow.keras.datasets import mnist


# d0 = mnist.load_data()
# # d0 = pd.d0
# # d0=np.asarray(d0).astype(np.int)
# # d0=np.asarray(d0).astype(np.float)



# import requests
# from urllib.request import Request, urlopen
# from bs4 import BeautifulSoup
# from urllib.parse import urlencode, quote_plus, unquote
# import json



# # 서비스URL
# url = d0

# df = pd.DataFrame(url)

# df.to_csv("../개인프로젝트/mnist.csv", mode='w',encoding='euc-kr')



# (x_train, y_train),(x_test,y_test) = mnist.load_data()

# # d0 = x_train

# d0 = np.append(x_train, x_test, axis=0)


d0 = pd.DataFrame(mnist.load_data())

l = d0['label']
d = d0.drop("label",axis=1)

print(d.shape)
print(l.shape)

# displaying the image
plt.figure(figsize=(5,5))
idx = 7
grid_data = d.iloc[idx].to_numpy().reshape(28,28)  
plt.imshow(grid_data, interpolation = "none", cmap = "gray")
plt.show()
print(l[idx])


#taking sample data
labels = l.head(35000)
data = d.head(35000)
print("Data shape = ", data.shape)
#output:
#Data shape =  (35000, 784)


from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
sc = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(data, labels, stratify=labels, test_size=0.22, random_state=42)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

clftre = XGBClassifier()
clftre.fit(X_train,y_train)

from sklearn.metrics import classification_report
print(classification_report(y_test, clftre.predict(X_test)))

#Using PCA technique
pca = PCA(svd_solver='randomized', random_state=42)
pca.fit(X_train)

#%matplotlib inline
fig = plt.figure(figsize = (12,8))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('n_components')
plt.ylabel('cumulative_explained_variance')
plt.show()


'''

parameters = [
    {"n_estimators":[100,200,300], "learning_rate":[0.1,0.3,0.001,0.01],"max_depth":[4,5,6]},
    {"n_estimators":[90,100,110], "learning_rate":[0.1, 0.001, 0.01],"max_depth":[4,5,6], "colsample_bytree":[0.6, 0.9, 1]},
    {"n_estimators":[90,110], "learning_rate":[0.1, 0.001, 0.5],"max_depth":[4,5,6], "colsample_bytree":[0.6, 0.9, 1],"colsample_bylevel":[0.6,0.7,0.9]}
]
# n_jobs = -1

import numpy as np
import time
from sklearn.decomposition import PCA
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28*28).astype('float32')/255
x_test = x_test.reshape(10000, 28*28).astype('float32')/255

# print(x_train.shape, x_test.shape)   # (60000, 784) (10000, 784)
# print(y_train.shape, y_test.shape)   # (60000, 10) (10000, 10)

pca = PCA(n_components=154)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

#2. 모델구성
model = GridSearchCV(XGBClassifier(), parameters, cv=kfold, verbose=1, refit=True, n_jobs = -1)

#3. 훈련
start = time.time()
model.fit(x_train, y_train)
end = time.time()

#4. 평가, 예측
result = model.score(x_test, y_test)    # score는 자동으로 맞춰서 반환해줌; 여기서 반환해주는건 'accuracy' (분류모델이기 때문에)

print("최적의 매개변수 : ", model.best_estimator_)   
print("최적의 파라미터 : ", model.best_params_)      

print("best_score_ : ", model.best_score_)            
print("model.score : ", model.score(x_test, y_test))  

y_predict = model.predict(x_test)
print("accuracy_score : ", accuracy_score(y_test, y_predict))   


y_pred_best = model.best_estimator_.predict(x_test)    
print("최적 튠 ACC : ", accuracy_score(y_test, y_pred_best))

print("time : ", end - start)
# print(model.feature_importances_)
'''









