from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import fetch_california_housing, load_boston, load_wine
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


#1.데이타
# datasets = fetch_california_housing()
# datasets = load_boston()
datasets = load_wine()

x = datasets.data
y = datasets['target']
print(x.shape, y.shape) #(20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=66, train_size=0.8)


scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test =scaler.transform(x_test)


#불러오기 (2모델,3훈련)
import pickle

model = pickle.load(open('./_save/m23_pickle1_save.dat', 'rb'))

#4. 평가
results = model.score(x_test, y_test) 
print("results :", results)
y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
print("r2 :", r2)

#####################
hist = model.evals_result()  #히스토리
print(hist)
