import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Bidirectional, Dropout, Flatten, Conv1D, Input
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, LabelEncoder, OneHotEncoder
import time 
def RMSE(y_test,y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

#1. 데이터 분석
path = "D:\\Study\\_data\\kaggle\\bike\\"
train = pd.read_csv(path + "train.csv") # (10886, 12)
test_file = pd.read_csv(path + "test.csv") # (6493, 9)
submit_file = pd.read_csv(path + "sampleSubmission.csv") # (6493, 2)

x = train.drop(columns=['datetime','workingday','holiday', 'casual', 'registered', 'count'], axis=1) 
y = train['count']
y = np.log1p(y) 
test_file = test_file.drop(columns=['datetime','workingday','holiday'], axis=1)



print(x.shape, y.shape)  #(10886, 7) (10886,)
# x = x.np.to_array()

# import matplotlib.pyplot as plt
# import seaborn as sns

# plt.figure(figsize=(8,8))
# sns.heatmap(data= x.corr(), square=True, annot=True, cbar=True)
# plt.show()   

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=66)



scaler = MinMaxScaler()
#scaler=StandardScaler()
#scaler=RobustScaler()
#scaler=MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) 
test_file = scaler.transform(test_file)

x_train = x_train.reshape(x_train.shape[0], 6, 1)
x_test = x_test.reshape(x_test.shape[0], 6, 1)
test_file = test_file.reshape(test_file.shape[0], 6, 1)

#2. 모델구성
model = Sequential()
model.add(Conv1D(25,2, activation='relu', input_shape=( 6, 1)))
model.add(Flatten()) ##
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1))



#3. 컴파일
model.compile(loss='mse', optimizer='adam')

# import datetime
# date = datetime.datetime.now()
# datetime = date.strftime("%m%d_%H%M") 

# filepath = './_ModelCheckPoint/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'      
# model_path = "".join([filepath, '7_kaggle_bike_', datetime, '_', filename])

es = EarlyStopping(monitor="val_loss", patience=20, mode='min', verbose=1, restore_best_weights=True)
#mcp = ModelCheckpoint(monitor='val_loss', mode='min', save_best_only=True, filepath=model_path)

start = time.time()

model.fit(x_train, y_train, epochs=1000, batch_size=25, verbose=2, validation_split=0.1, callbacks=[es])#, mcp])
end = time.time()- start
print("걸린시간 : ", round(end, 3), '초')




#4. 평가, 예측
loss = model.evaluate (x_test, y_test)
print('loss :', loss) #loss :
y_pred = model.predict(test_file)


# rmse = RMSE(y_test, y_pred)
# print('RMSE :', rmse) #loss :

# r2 = r2_score(y_test, y_pred)
# print('r2score :', r2) 


###################제출용 제작############################
results = model.predict(test_file)

print(results.shape)
results = np.expm1(results)
submit_file['count'] = results.astype(int)

print(submit_file[:10])

submit_file.to_csv(path + 'submit_1216_05.csv', index = False)


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