import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Bidirectional, Dropout, Flatten, Conv1D
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, LabelEncoder, OneHotEncoder
import time 

#1. 데이터 분석
path = "D:\\Study\\_data\\kaggle\\bike\\"
train = pd.read_csv(path + "train.csv") # (10886, 12)
test_file = pd.read_csv(path + "test.csv") # (6493, 9)
submit_file = pd.read_csv(path + "sampleSubmission.csv") # (6493, 2)

x = train.drop(columns=['datetime', 'casual', 'registered', 'count'], axis=1) 
y = train['count']
y = np.log1p(y) 
test_file = test_file.drop(columns=['datetime'], axis=1)

print(x.shape, y.shape)  #(10886, 8) (10886,)
#x = x.np.to_array()


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)



#scaler = MinMaxScaler()
#scaler=StandardScaler()
#scaler=RobustScaler()
scaler=MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) 

x_train = x_train.reshape(x_train.shape[0], 8, 1)
x_test = x_test.reshape(x_test.shape[0], 8, 1)

#2. 모델구성
model = Sequential()
model.add(Conv1D(250,2, input_shape=(8,1)))
model.add(Flatten()) ##
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))



#3. 컴파일
model.compile(loss='mse', optimizer='adam')

# import datetime
# date = datetime.datetime.now()
# datetime = date.strftime("%m%d_%H%M") 

# filepath = './_ModelCheckPoint/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'      
# model_path = "".join([filepath, '7_kaggle_bike_', datetime, '_', filename])

es = EarlyStopping(monitor="val_loss", patience=10, mode='min', verbose=1, restore_best_weights=True)
#mcp = ModelCheckpoint(monitor='val_loss', mode='min', save_best_only=True, filepath=model_path)

start = time.time()

model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=2, validation_split=0.2, callbacks=[es])#, mcp])
end = time.time()- start
print("걸린시간 : ", round(end, 3), '초')

#4. 결과
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_pred = model.predict(x_test)



'''
기존
loss : 1.4842497110366821
---------------------
LSTM 반영시 값이 좋아졌다
걸린시간 :  17.283 초
loss :  1.4496670961380005
=====================
Conv1D
걸린시간 :  2.724 초
loss :  1.4772008657455444



# from sklearn.metrics import r2_score 
# r2 = r2_score(y_test, y_pred)
# print("r2스코어", r2)

# def RMSE(y_test, y_pred): 
#     return np.sqrt(mean_squared_error(y_test, y_pred))   
# rmse = RMSE(y_test, y_pred) 
# print("RMSE : ", rmse)

'''