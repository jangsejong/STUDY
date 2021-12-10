import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import RobustScaler, MaxAbsScaler, LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
import sys

#1 데이터
path = "D:\\Study\\_data\\dacon\\wine\\"
train = pd.read_csv(path +"train.csv")
test_file = pd.read_csv(path + "test.csv") 
submission = pd.read_csv(path+"sample_Submission.csv") #제출할 값

# print(train)

y = train['quality']
x = train.drop(['id','quality'], axis =1)
# x = x.drop(['citric acid','pH','sulphates']
le = LabelEncoder()
le.fit(train['type'])
x['type'] = le.transform(train['type'])

# y = np.array(y).reshape(-1,1)
# enc= OneHotEncoder()   #[0. 0. 1. 0. 0.]
# enc.fit(y)
# y = enc.transform(y).toarray()

y = y.to_numpy()
x = x.to_numpy()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
         train_size = 0.8, shuffle = True, random_state = 66) #455.2 /114

scaler = RobustScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_validate

rf = RandomForestClassifier(n_jobs = -1 , random_state =66)
score = cross_validate(rf, x_train, y_train, return_train_score=True, n_jobs= -1)
print(np.mean(score['train_score']), np.mean(score['test_score'])) #1.0 0.4582002608969457

rf.fit(x_train, y_train)
print(rf.feature_importances_) #[0.07849329 0.10240398 0.07930172 0.08278457 0.08542998 0.08759306 0.09092311 0.10002204 0.07846882 0.08340361 0.12774785 0.00342798]

rf = RandomForestClassifier(oob_score= True, n_jobs = -1 , random_state =66)
rf.fit(x_train, y_train)
print(rf.oob_score_) #0.8517027863777089 ->>>>>>>>>>>>>

es = ExtraTreesClassifier(n_jobs = -1 , random_state =66)
#score = cross_validate(es, x_train, y_train, return_train_score=True, n_jobs= -1)
#print(np.mean(score['train_score']), np.mean(score['test_score'])) 

gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.2, random_state=66)
score = cross_validate(gb, x_train, y_train, return_train_score=True, n_jobs= -1)
print(np.mean(score['train_score']), np.mean(score['test_score'])) #1.0 0.4574310647294319