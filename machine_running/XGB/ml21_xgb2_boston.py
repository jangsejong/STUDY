from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import fetch_california_housing, load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import RobustScaler, MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer  # https://tensorflow.blog/2018/01/14/quantiletransformer/
from sklearn.preprocessing import PowerTransformer      #https://wikidocs.net/83559
from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import learning_curve, train_test_split
from sklearn.metrics import r2_score, accuracy_score
import time
import warnings
warnings.filterwarnings("ignore")


#1.데이타
# datasets = fetch_california_housing()
datasets = load_boston()

x = datasets.data
y = datasets['target']
print(x.shape, y.shape) #(20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=66, train_size=0.8)


scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test =scaler.transform(x_test)


#2. 모델
model = XGBRegressor(
    n_jobs = -1,
    n_estimators=2000,
    learning_rate = 0.025,
    max_depth = 4,
    min_child_weight = 1,
    subsample =1,
    colsample_bytree =1,
    reg_alpha =1,               #규제 L1
    reg_lambda=0,               #규제 L2
    
)

#3. 훈련
start = time.time()
model.fit(x_train, y_train, verbose=1)
end = time.time()

print( "걸린시간 :", end - start)


#4. 결과
results = model.score(x_test, y_test) 
print("results :", round(results,4))
y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
print("r2 :", round(r2, 4))


'''
load_boston
걸린시간 : 1.0122942924499512
0.9313449710981906
r2 : 0.9313449710981906

fetch_california_housing
걸린시간 : 18.054712057113647
0.8566291699938181
r2 : 0.8566291699938181
'''