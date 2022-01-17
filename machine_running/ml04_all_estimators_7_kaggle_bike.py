from sklearn.utils import all_estimators #회기 R2 분류 acc
from sklearn.metrics import accuracy_score
from sklearn import datasets
import warnings
warnings.filterwarnings('ignore')


import numpy as np
import pandas as pd
from tensorflow.python.keras.metrics import accuracy



#1. 데이터 분석
path = "D:\\Study\\_data\\kaggle\\bike\\"
train = pd.read_csv(path + "train.csv") # (10886, 12)
test_file = pd.read_csv(path + "test.csv") # (6493, 9)
submit_file = pd.read_csv(path + "sampleSubmission.csv") # (6493, 2)

x = train.drop(columns=['datetime', 'casual', 'registered', 'count'], axis=1) 
y = train['count']
# y = np.log1p(y) 
test_file = test_file.drop(columns=['datetime'], axis=1)

# #1. 데이터
# datasets= load_iris()
# #print(datasets.DESCR)
# x = datasets.data
# y = datasets.target


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, shuffle=True, random_state=66)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#모델 구성

allAlgorithms = all_estimators(type_filter='classifier')
# allAlgorithms = all_estimators(type_filter='regressor')

print("allAlgorithms :", allAlgorithms)
print("모델의 갯수 :", len(allAlgorithms))


for (name, algorithm) in allAlgorithms:
  try: 
    model = algorithm()
    model.fit(x_train, y_train)
        
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(name, '의 정답율 :', acc)
    
  except:
    continue



'''

'''