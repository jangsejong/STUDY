from soupsieve import select
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import fetch_california_housing, fetch_covtype, load_boston, load_wine, load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import RobustScaler, MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer  # https://tensorflow.blog/2018/01/14/quantiletransformer/
from sklearn.preprocessing import PowerTransformer      #https://wikidocs.net/83559
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.metrics import f1_score, r2_score, accuracy_score, mean_squared_error
import time
import numpy as np
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings("ignore")


#1.데이터 로드
# datasets = fetch_california_housing()
# datasets = load_boston()
# datasets = load_wine()
# datasets = fetch_covtype()

x, y = load_boston(return_X_y=True)
print(x.shape, y.shape)





from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import RobustScaler, MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer  # https://tensorflow.blog/2018/01/14/quantiletransformer/
from sklearn.preprocessing import PowerTransformer      #https://wikidocs.net/83559
from sklearn.preprocessing import PolynomialFeatures
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = RobustScaler()
# scaler = MaxAbsScaler()
# scaler = QuantileTransformer()
# scaler = PowerTransformer()
# scaler = PolynomialFeatures()
# scaler.fit(x)
# x1 = scaler.transform(x)
# y =scaler.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=66, train_size=0.8)


#2. 모델
model = XGBRegressor(
    n_jobs = -1,
    n_estimators=2000,
    learning_rate = 0.4,
    max_depth = 6,
    min_child_weight = 0.9,
    subsample =1,
    colsample_bytree =0.9,
    reg_alpha =1,               #규제 L1
    reg_lambda=0,               #규제 L2
    tree_method= 'gpu_hist',
    predictor= 'gpu_predictor',
)

#3. 훈련
start = time.time()
model.fit(x_train, y_train, verbose=100,
          eval_set=[(x_test, y_test)],
          eval_metric='rmse',           #logloss, error, rmse, mse
          early_stopping_rounds=100
          )
end = time.time()

print( "걸린시간 :", end - start)







#4. 평가
score = model.score(x_test, y_test) 
print("score :", score)


#####################
# hist = model.evals_result()  #히스토리
# print(hist)
model.feature_importances_
print(np.sort(model.feature_importances_))
aaa = np.sort(model.feature_importances_)
print(aaa)

print("==============================================")

for thresh in aaa:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    selet_x_train = selection.transform(x_train)
    selet_x_test = selection.transform(x_test)
    print(selet_x_train.shape, selet_x_test.shape)
    
    selection_model = XGBRegressor(n_jobs=-1)
    selection_model.fit(selet_x_train, y_train)
    
    y_predict = selection_model.predict(selet_x_test)
    score = r2_score(y_test, y_predict)
    
    print("Thresh=%.3f, n=%d, R2: %.2f%%"
          %(thresh, selet_x_train.shape[1], score*100))


