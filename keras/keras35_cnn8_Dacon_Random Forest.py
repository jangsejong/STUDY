import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import RobustScaler, MaxAbsScaler, LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt



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

rf = RandomForestClassifier(n_jobs = -1 ,n_estimators=5000, random_state =66)
score = cross_validate(rf, x_train, y_train, return_train_score=True, n_jobs= -1)
print("RandomForestClassifier")
print(np.mean(score['train_score']), np.mean(score['test_score'])) #1.0 0.4582002608969457

rf = RandomForestClassifier(oob_score= True, n_jobs = -1 , random_state =66)
rf.fit(x_train, y_train)
print("RandomForestClassifier-oob")
print(rf.oob_score_) #0.8517027863777089 ->>>>>>>>>>>>>

es = ExtraTreesClassifier(n_estimators=5000, random_state =66)
score = cross_validate(es, x_train, y_train, return_train_score=True, n_jobs= -1)
print("ExtraTreesClassifier")
print(np.mean(score['train_score']), np.mean(score['test_score'])) 

gb = GradientBoostingClassifier(n_estimators=5000, learning_rate=0.2, random_state=66)
score = cross_validate(gb, x_train, y_train, return_train_score=True, n_jobs= -1)
print("GradientBoostingClassifier")
print(np.mean(score['train_score']), np.mean(score['test_score'])) #1.0 0.5770140794386218

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
hgb = HistGradientBoostingClassifier(n_estimators=5000,random_state =66)
score = cross_validate(hgb, x_train, y_train, return_train_score=True, n_jobs= -1)

print("HistGradientBoostingClassifier")
print(np.mean(score['train_score']), np.mean(score['test_score'])) 
hgb.fit(x_test, y_test)
print(hgb.score(x_test, y_test))

#from sklearn.inspection import permutation_importance
#hgb.fit(x_train, y_train)
#result = permutation_importance(hgb,x_train, y_train, n_repeats=10, random_state=66, n_jobs=-1)

from lightgbm import LGBMClassifier

lgb = LGBMClassifier(n_estimators=5000,random_state =66)
score = cross_validate(lgb, x_train, y_train, return_train_score=True, n_jobs= -1)
print("LGBMClassifier")
print(np.mean(score['train_score']), np.mean(score['test_score']))
