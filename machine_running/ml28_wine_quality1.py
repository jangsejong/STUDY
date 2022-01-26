from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import fetch_california_housing, fetch_covtype, load_boston, load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import RobustScaler, MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer  # https://tensorflow.blog/2018/01/14/quantiletransformer/
from sklearn.preprocessing import PowerTransformer      #https://wikidocs.net/83559
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.metrics import f1_score, r2_score, accuracy_score
import time
import numpy as np
import warnings
warnings.filterwarnings("ignore")


#1.데이터 로드
# datasets = fetch_california_housing()
# datasets = load_boston()
# datasets = load_wine()
# datasets = fetch_covtype()

path = 'D:\\Study\\_data\\dacon\\whitewine\\'
# datasets = pd.read_csv(path +"winequality-white.csv", thousands=",", encoding='cp949',sep=';')
datasets = pd.read_csv(path + "winequality-white.csv",delimiter=';')

print(datasets.shape) #(4898, 12)

'''
x = datasets[ :, :11]
y = datasets[ :, 11]
'''


# print(datasets.isnull().sum())

import matplotlib.pyplot as plt


def boxplot_vis(data, target_name):
    plt.figure(figsize=(30, 30))
    for col_idx in range(len(data.columns)):
        # 6행 2열 서브플롯에 각 feature 박스플롯 시각화
        plt.subplot(6, 2, col_idx+1)
        # flierprops: 빨간색 다이아몬드 모양으로 아웃라이어 시각화
        plt.boxplot(data[data.columns[col_idx]], flierprops = dict(markerfacecolor = 'r', marker = 'D'))
        # 그래프 타이틀: feature name
        plt.title("Feature" + "(" + target_name + "):" + data.columns[col_idx], fontsize = 20)
    # plt.savefig('../figure/boxplot_' + target_name + '.png')
    # plt.show()
boxplot_vis(datasets,'white_wine')

def remove_outlier(input_data):
    q1 = input_data.quantile(0.25) # 제 1사분위수
    q3 = input_data.quantile(0.75) # 제 3사분위수
    iqr = q3 - q1 # IQR(Interquartile range) 계산
    minimum = q1 - (iqr * 1.5) # IQR 최솟값
    maximum = q3 + (iqr * 1.5) # IQR 최댓값
    # IQR 범위 내에 있는 데이터만 산출(IQR 범위 밖의 데이터는 이상치)
    df_removed_outlier = input_data[(minimum < input_data) & (input_data < maximum)]
    return df_removed_outlier

# 이상치 제거한 데이터셋
prep = remove_outlier(datasets)

# 목표변수 할당
prep['target'] = 0

# 결측치(이상치 처리된 데이터) 확인
a = prep.isnull().sum()
print(a)

# 이상치 포함 데이터(이상치 처리 후 NaN) 삭제
prep.dropna(axis = 0, how = 'any', inplace = True)
print(f"이상치 포함된 데이터 비율: {round((len(datasets) - len(prep))*100/len(datasets), 2)}%")


x = prep.drop('quality', axis=1)
y = prep['quality']

print(x.shape, y.shape) #(3841, 12) (3841,)

scaler = StandardScaler()
scaler.fit(x)
x1 = scaler.transform(x)
# y =scaler.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x1, y, shuffle=True, random_state=66, train_size=0.8)




# #data save
# import pickle
# pickle.dump(datasets, open('./_save/m26_pickle1_save.dat', 'wb'))

# import pickle
# datasets = pickle.load(open('./_save/m26_pickle1_save.dat', 'rb'))

# x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=66, train_size=0.8)


#2. 모델
model = XGBClassifier(
    n_jobs = -1,
    n_estimators=10000,
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
          eval_metric='mlogloss',           #logloss, error, rmse, mse
          early_stopping_rounds=30
          )
end = time.time()

print( "걸린시간 :", end - start)







#4. 평가
results = model.score(x_test, y_test) 
print("results :", results)
y_pred = model.predict(x_test)

acc = accuracy_score(y_test, y_pred)
print("acc :", acc)
print('f1_score :', f1_score(y_test, y_pred, average='micro'))
print('f1_score :', f1_score(y_test, y_pred, average='macro'))

#####################
# hist = model.evals_result()  #히스토리
# print(hist)
