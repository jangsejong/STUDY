import pandas as pd
import numpy as np
from keras.layers.core import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential, Model
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint  
from tensorflow.keras. optimizers import SGD
from keras import optimizers
from sklearn import metrics
from tensorflow.python.keras.backend import relu
from tensorflow.python.keras.layers.core import Activation
from sklearn import metrics

def f1_score(answer, submission):
    true = answer
    pred = submission
    score = metrics.f1_score(y_true=true, y_pred=pred)
    return score

path = "D:\\Study\\_data\\dacon\\heart\\"
train = pd.read_csv(path + "train.csv")
test_file = pd.read_csv(path + "test.csv")
submit_file = pd.read_csv(path + "sample_submission.csv")


# print(train.shape)  #(151, 15)
# print(test_file.shape)  #(152, 14)
# print(submit_file.shape) #(152, 2)

# print(train.describe) #(151, 15)

#        id  age  sex  cp  trestbps  chol  fbs  restecg  thalach  exang  oldpeak  slope  ca  thal  target
# 0      1   53    1   2       130   197    1        0      152      0      1.2      0   0     2       1
# 1      2   52    1   3       152   298    1        1      178      0      1.2      1   0     3       1
# 2      3   54    1   1       192   283    0        0      195      0      0.0      2   1     3       0
# 3      4   45    0   0       138   236    0        0      152      1      0.2      1   0     2       1
# 4      5   35    1   1       122   192    0        1      174      0      0.0      2   0     2       1
# ..   ...  ...  ...  ..       ...   ...  ...      ...      ...    ...      ...    ...  ..   ...     ...
# 146  147   50    1   2       140   233    0        1      163      0      0.6      1   1     3       0
# 147  148   51    1   2        94   227    0        1      154      1      0.0      2   1     3       1
# 148  149   69    1   3       160   234    1        0      131      0      0.1      1   1     2       1
# 149  150   46    1   0       120   249    0        0      144      0      0.8      2   0     3       0
# 150  151   63    0   1       140   195    0        1      179      0      0.0      2   2     2       1

# print(test_file.describe) #(152, 14)
#        id  age  sex  cp  trestbps  chol  fbs  restecg  thalach  exang  oldpeak  slope  ca  thal
# 0      1   57    1   0       150   276    0        0      112      1      0.6      1   1     1
# 1      2   59    1   3       170   288    0        0      159      0      0.2      1   0     3
# 2      3   57    1   2       150   126    1        1      173      0      0.2      2   1     3
# 3      4   56    0   0       134   409    0        0      150      1      1.9      1   2     3
# 4      5   71    0   2       110   265    1        0      130      0      0.0      2   1     2
# ..   ...  ...  ...  ..       ...   ...  ...      ...      ...    ...      ...    ...  ..   ...
# 147  148   64    0   0       130   303    0        1      122      0      2.0      1   2     2
# 148  149   43    0   0       132   341    1        0      136      1      3.0      1   0     3
# 149  150   53    1   0       123   282    0        1       95      1      2.0      1   2     3
# 150  151   67    0   2       152   277    0        1      172      0      0.0      2   1     2
# 151  152   43    0   2       122   213    0        1      165      0      0.2      1   0     2

# print(submit_file.describe) #(152, 2)

# [152 rows x 14 columns]>
#        id  target
# 0      1      -1
# 1      2      -1
# 2      3      -1
# 3      4      -1
# 4      5      -1
# ..   ...     ...
# 147  148      -1
# 148  149      -1
# 149  150      -1
# 150  151      -1
# 151  152      -1

# print(train.columns)
# Index(['id', 'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
#        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'],
#       dtype='object')
# print(test_file.columns)
# Index(['id', 'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
#        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'],
#       dtype='object')
# print(submit_file.columns)
# Index(['id', 'target'], dtype='object')

x = train.drop(['id','target'], axis=1)#.drop(index=131)  #컬럼 삭제할때는 드랍에 액시스 1 준다   
#print(x.shape) (151, 13)
test_file=test_file.drop(['id'], axis=1)
#print(test_file.shape) (152, 13)
y = train['target']#.drop(index=131)
#print(y.shape) (151,1)

x = x.to_numpy()
y = y.to_numpy()
test_file = test_file.to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=66)

model = Sequential()
model.add(Dense(32, activation='linear', input_dim=13))
#model.add(Dense(30, activation='relu'))
model.add(Dense(16, activation='linear'))
#model.add(Dense(6, activation='linear'))
model.add(Dense(4, activation='linear'))
#model.add(Dense(2, activation='linear'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


############################################################################################################
import datetime
date=datetime.datetime.now()   
datetime = date.strftime("%m%d_%H%M")   #1206_0456
#print(datetime)

filepath='./_ModelCheckPoint/'
filename='{epoch:04d}-{val_loss:.4f}.hdf5'  # 2500-0.3724.hdf
model_path = "".join([filepath, 'k26_', datetime, '_', filename])
            #./_ModelCheckPoint/k26_1206_0456_2500-0.3724.hdf
##############################################################################################################

es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)

mcp=ModelCheckpoint(monitor='val_loss', mode='max', verbose=1, 
                    save_best_only=True,
                    filepath=model_path)

model.fit(x_train, y_train, epochs=1000, batch_size=13, validation_split=0.1, callbacks=[es, mcp])

#scaler = MinMaxScaler()
scaler=StandardScaler()
#scaler=RobustScaler()
#scaler=MaxAbsScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
test_file=scaler.transform(test_file)

loss = model.evaluate(x_test, y_test)
y_predict=model.predict(x_test)
y_predict=y_predict.round(0).astype(int)
f1=f1_score(y_test, y_predict)
print('loss : ',loss[0])
print('accuracy :  ', loss[1])
print('f1_score :  ', f1)

results=model.predict(test_file)
results=results.round(0).astype(int)

submit_file['target']=results
submit_file.to_csv(path + "heart_ann009.csv", index=False)  
'''
001
loss :  0.30531740188598633
accuracy :   0.875
f1_score :   0.9

002
loss :  0.39412742853164673
accuracy :   0.9375
f1_score :   0.9473684210526316

003
loss :  0.28279146552085876
accuracy :   0.9375
f1_score :   0.9473684210526316

004
loss :  0.22273194789886475
accuracy :   0.9375
f1_score :   0.9473684210526316

005
loss :  0.23493283987045288
accuracy :   0.9375
f1_score :   0.9473684210526316

006  0.1/13
loss :  0.20334351062774658
accuracy :   0.9375
f1_score :   0.9473684210526316

007
loss :  0.10557577013969421
accuracy :   0.9375
f1_score :   0.9473684210526316

008
loss :  0.15957488119602203
accuracy :   1.0
f1_score :   1.0

009
loss :  0.08917468041181564
accuracy :   0.9375
f1_score :   0.9473684210526316

'''