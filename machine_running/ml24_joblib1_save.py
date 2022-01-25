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


#2. 모델
model = XGBRegressor(
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
          eval_metric='rmse',           #logloss, error, rmse, mse
          early_stopping_rounds=2000
          )
end = time.time()

print( "걸린시간 :", end - start)


#4. 결과
results = model.score(x_test, y_test) 
print("results :", round(results,4))
y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
print("r2 :", round(r2, 4))


#####################
# hist = model.evals_result()
# print(hist)

#저장
import joblib
joblib.dump(model, './_save/m23_pickle2_save.dat')



# from catboost import CatBoostClassifier, Pool, sum_models
# import shap
# import matplotlib.pyplot as plt
# from matplotlib.ticker import Formatter
# import plotly.graph_objects as go
# import xgboost
# import shap

# # # explain the model's predictions using SHAP
# # explainer = shap.Explainer(model)
# # shap_values = explainer(x)

# # # visualize the first prediction's explanation
# # shap.plots.waterfall(shap_values[0])

# loss = hist.get('validation_0').get('rmse')
# epochs = range(1, len(hist.get('validation_0').get('rmse')) + 1)
    
# plt.plot(epochs, loss, 'y--', label="training loss")

# plt.grid()
# plt.legend()
# plt.show()
'''
results = model.evals_result()

import matplotlib.pyplot as plt

train_error = results['validation_0']['rmse']
test_error = results['validation_1']['rmse']

epoch = range(1, len(train_error)+1)
plt.plot(epoch, train_error, label = 'Train')
plt.plot(epoch, test_error, label = 'Test')
plt.ylabel('Classification Error')
plt.xlabel('Model Complexity (n_estimators)')
plt.legend()
plt.show()


걸린시간 : 0.9604544639587402
results : 0.9796
r2 : 0.9796
'''
'''
load_boston
걸린시간 : 1.0122942924499512
0.9313449710981906
r2 : 0.9313449710981906

fetch_california_housing
걸린시간 : 18.054712057113647
0.8566291699938181
r2 : 0.8566291699938181

results : 0.8455
r2 : 0.8455

'''