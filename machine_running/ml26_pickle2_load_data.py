from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import fetch_california_housing, fetch_covtype, load_boston, load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import RobustScaler, MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer  # https://tensorflow.blog/2018/01/14/quantiletransformer/
from sklearn.preprocessing import PowerTransformer      #https://wikidocs.net/83559
from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import learning_curve, train_test_split
from sklearn.metrics import r2_score, accuracy_score
import time
# import warnings
# warnings.filterwarnings("ignore")


# #1.데이타
# # datasets = fetch_california_housing()
# # datasets = load_boston()
# # datasets = load_wine()
# datasets = fetch_covtype()

# x = datasets.data
# y = datasets['target']
# print(x.shape, y.shape) #

# x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=66, train_size=0.8)


# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test =scaler.transform(x_test)

# #data save
# import pickle
# pickle.dump(datasets, open('./_save/m26_pickle1_save.dat', 'wb'))

import pickle
datasets = pickle.load(open('./_save/m26_pickle1_save.dat', 'rb'))
x = datasets.data
y = datasets['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=66, train_size=0.8)


#2. 모델
model = XGBClassifier(
    n_jobs = -1,
    n_estimators=2000,
    learning_rate = 0.095,
    max_depth = 6,
    min_child_weight = 0.9,
    subsample =1,
    colsample_bytree =0.9,
    reg_alpha =1,               #규제 L1
    reg_lambda=0,               #규제 L2
    
)

#3. 훈련
start = time.time()
model.fit(x_train, y_train, verbose=10,
          eval_set=[(x_test, y_test)],
          eval_metric='mlogloss',           #logloss, error, rmse, mse
          early_stopping_rounds=2000
          )
end = time.time()

print( "걸린시간 :", end - start)







#4. 평가
results = model.score(x_test, y_test) 
print("results :", results)
y_pred = model.predict(x_test)

acc = accuracy_score(y_test, y_pred)
print("acc :", acc)

#####################
hist = model.evals_result()  #히스토리
print(hist)
