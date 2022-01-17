import tensorflow as tf
import seaborn as sns

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
from tensorflow.python.keras.backend import relu
from tensorflow.python.keras.layers.core import Activation
from sklearn import metrics
import matplotlib.pyplot as plt


def f1_score(answer, submission):
    true = answer
    pred = submission
    score = metrics.f1_score(y_true=true, y_pred=pred)
    return score

path = "D:\\Study\\_data\\dacon\\heart\\"
train = pd.read_csv(path + "train.csv")
test_file = pd.read_csv(path + "test.csv")
submit_file = pd.read_csv(path + "sample_submission.csv")

feature_columns = []
for header in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']:
    feature_columns.append(tf.feature_column.numeric_column(header))
    
age = tf.feature_column.numeric_column("age")
age_buckets = tf.feature_column.bucketized_column(age, boundaries=[18, 25,30,35,40,45,50,55,60,65,70,75,77])
feature_columns.append(age_buckets)

# train["thal"] = train["thal"].apply(str)
thal = tf.feature_column.categorical_column_with_vocabulary_list('thal',['0','1','2','3'])
# htal_one_hot = tf.feature_column.indicator_column(thal)
# feature_columns.append(htal_one_hot)

# train['slope'] = train['slope'].apply(str)
# slope = tf.feature_column.sequence_categorical_column_with_vocabulary_list('slope',['0','1','2'])
# slope_one_hot = tf.feature_column.indicator_column(slope)
# feature_columns.append(slope_one_hot)

# test_file["thal"] = test_file["thal"].apply(str)
# thal = tf.feature_column.categorical_column_with_vocabulary_list('thal',['3','6','7'])
# htal_one_hot = tf.feature_column.indicator_column(thal)
# feature_columns.append(htal_one_hot)

thal_embedding = tf.feature_column.embedding_column(thal, dimension=2)
feature_columns.append(thal_embedding)

age_thai_crossed = tf.feature_column.crossed_column([age_buckets,thal],hash_bucket_size=1000)
age_thai_crossed = tf.feature_column.indicator_column(age_thai_crossed)
feature_columns.append(age_thai_crossed)

#cp_slope_crossed = tf.feature_column.crossed_column([cp,slope],hash_bucket_size=1000)

# print(train.shape)  #(151, 15)
# print(test_file.shape)  #(152, 14)
# print(submit_file.shape) #(152, 2)
# print(train.describe) #(151, 15)

x = train.drop(['id','target','chol','fbs','restecg','trestbps'], axis =1)
y = train['target']

#print(train.shape, test_file.shape)           # (151, 15) (152, 14)

# x = x.drop([''],axis =1)
test_file =test_file.drop(['id','chol','fbs','restecg','trestbps'],axis =1)
test_file.iloc[40,7] = "3"
test_file.iloc[45,7] = "3"
test_file.iloc[79,7] = "3"
test_file.iloc[80,7] = "3"
test_file.iloc[95,7] = "3"
#print(test_file)


# print(x.shape, test_file.shape)  (151, 10) (152, 10)

y = y.to_numpy()
x = x.to_numpy()
test_file = test_file.to_numpy()



x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=66)

scaler = MinMaxScaler()#feature_range=(0,100))
#scaler=StandardScaler()
#scaler=RobustScaler()
#scaler=MaxAbsScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
test_file=scaler.transform(test_file)

model = Sequential()
model.add(Dense(300, activation='relu', input_dim=9))
model.add(Dropout(0.3))
# model.add(Dense(80, activation='relu'))
model.add(Dense(300, activation='relu'))
# # model.add(Dense(32, activation='linear'))
model.add(Dropout(0.1))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.1))
# model.add(Dense(4, activation='linear'))
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

es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, restore_best_weights=True)

mcp=ModelCheckpoint(monitor='val_loss', mode='max', verbose=1, 
                    save_best_only=True,
                    filepath=model_path)

model.fit(x_train, y_train, epochs=1000, batch_size=9, validation_split=0.1, callbacks=[es, mcp])



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
submit_file.to_csv(path + "heart_1223_6.csv", index=False)  

'''
heart_1222_2.csv
loss :  0.3836241662502289

heart_1222_3
loss :  0.43839049339294434
accuracy :   0.9677419066429138
f1_score :   0.9767441860465117

heart_1223_1
loss :  0.3176395297050476
accuracy :   0.9375
f1_score :   0.9473684210526316

heart_1223_2
loss :  0.31535929441452026
accuracy :   1.0
f1_score :   1.0

heart_1223_3
loss :  0.3062933087348938
accuracy :   1.0
f1_score :   1.0

heart_1223_4
loss :  0.5404765009880066
accuracy :   1.0
f1_score :   1.0

heart_1223_5
loss :  0.1639932096004486
accuracy :   0.9375
f1_score :   0.9473684210526316

heart_1223_6
loss :  0.21505576372146606
accuracy :   1.0
f1_score :   1.0

heart_1223_7
loss :  0.17381486296653748
accuracy :   1.0
f1_score :   1.0

heart_1223_8
loss :  0.12672555446624756
accuracy :   1.0
f1_score :   1.0

heart_1223_9
loss :  0.15368671715259552
accuracy :   0.9375
f1_score :   0.9473684210526316

heart_1223_10
loss :  0.16377535462379456
accuracy :   1.0
f1_score :   1.0

heart_1223_11
loss :  0.18633806705474854
accuracy :   0.9375
f1_score :   0.9473684210526316
'''

