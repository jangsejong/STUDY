import xgboost
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
import pandas as pd
import pandas as np
from xgboost.sklearn import XGBRegressor
import warnings
warnings.filterwarnings('ignore')
from matplotlib import font_manager, rc 
from xgboost import plot_importance
import seaborn as sns
import matplotlib
import platform


#matplotlib 에서 사용하는 폰트를 한글 지원이 가능한 것으로 바꾸는 코드
if platform.system() == 'Windows':
# 윈도우인 경우
    font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
    rc('font', family=font_name)
else:    
# Mac 인 경우
    rc('font', family='AppleGothic')
    
matplotlib.rcParams['axes.unicode_minus'] = False   
#그래프에서 마이너스 기호가 표시되도록 하는 설정입니다. 

#데이터 로드

path = 'D:\\Study\\_data\\dacon\\movie_review\\dataset\\'

# covid_19 = pd.read_csv(path +"코로나바이러스감염증-19_확진환자_발생현황_220107.csv", thousands=",", encoding='cp949')
train = pd.read_csv(path +"train.csv", thousands=",")#, encoding='cp949')
test = pd.read_csv(path +"test.csv", thousands=",")#, encoding='cp949')

# print(train.head())
# print(test.head())
print(train.info())
print(test.info())
print(train.shape) #(5000, 3)
print(test.shape)  #(5000, 2)

# DataSet search

# for col1 in train.columns:
#     n_nan1 = train[col1].isnull().sum()
#     if n_nan1>0:
#       msg1 = '{:^20}에서 결측치 개수 : {}개'.format(col1,n_nan1)
#       print(msg1)
#     else:
#           print('결측치가 없습니다.')    
# for col2 in test.columns:
#     n_nan2 = test[col2].isnull().sum()
#     if n_nan2>0:
#       msg2 = '{:^20}에서 결측치 개수 : {}개'.format(col2,n_nan2)
#       print(msg2)
#     else:
#           print('결측치가 없습니다.')  

# 결측치가 없다.
'여행에 집중할수 있게 편안한 휴식을 제공하는 호텔이었습니다. 위치선정 또한 적당한 편이었고 청소나 청결상태도 좋았습니다.'
'여행에 집중할수 있게 편안한 휴식을 제공하는 호텔이었습니다. 위치선정 또한 적당한 편이었고 청소나 청결상태도 좋았습니다.'



#정규 표현식 적용
import re

def apply_regular_expression(document):
    hangul = re.compile('[^ ㄱ-ㅣ 가-힣]')  # 한글 추출 규칙: 띄어 쓰기(1 개)를 포함한 한글
    result = hangul.sub('', document)  # 위에 설정한 "hangul"규칙을 "text"에 적용(.sub)시킴
    return result
print(train['document'][1])
print(apply_regular_expression(train['document'][1]))
#정규 표현식을 적용한 후 특수 문자가 잘 제거된 것을 확인할 수 있습니다.

#한국어 형태소 분석 - 명사 단위
from konlpy.tag import Kkma, Komoran, Okt, Mecab
from collections import Counter


mec = Mecab()
okt = Okt()
kkm = Kkma()
kom = Komoran()  # 명사 형태소 추출 함수
nouns = mec.nouns(apply_regular_expression(train['text'][1]))
print(nouns)




'''
#covid_19 데이터와 상관 관계를 위하여 날짜를 맞추기 위해 kospi 데이터 1월19일까지의 데이터를 삭제 해주었다.
kospi = kospi.drop(kospi.index[range(0,12)] ,axis=0)

# print(covid_19.info())
# print(kospi.head())


# 기존 일자를 new_data 로 수정후 일자 삭제, new_data를 인덱스로 넣어 주었다.
# covid_19['new_Date'] = pd.to_datetime(covid_19['일자'])
kospi['new_Date'] = pd.to_datetime(kospi['일자'])

# covid_19.drop('일자', axis = 1, inplace=True)
# covid_19.set_index('new_Date', inplace=True)

kospi.drop('일자', axis = 1, inplace=True)
kospi.set_index('new_Date', inplace=True)
# print(x1.head())
# print('\n')
# print(x1.info())
# print('\n')
# print(type(x1['new_Date'][1]))

            
# x1 = covid_19.drop(columns=[], axis=1) 
x2 = kospi.drop(columns =['현재지수'], axis=1) 
y2 = kospi['현재지수']


#print(x2.shape) #(484, 11)
# x22 = np.array(x2)
# y22 = np.array(y2)





df4_x_train, df4_x_test, df4_y_train, df4_y_test=train_test_split(x2, y2, test_size=0.3, random_state=66, shuffle=True)
xgb_model=xgboost.XGBRegressor(n_estimators=10000, learning_rate=0.08, gamma=0, subsampel=0.7, colsample_bytree=1, max_depth=7)

print(len(df4_x_train), len(df4_x_test))
xgb_model.fit(df4_x_train, df4_y_train)

XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1, colsample_bytree=1, gamma=0, learning_rate=0.08, max_delta_step=0, max_depth=7, min_child_weight=1,
             missing=None, n_estimators=100, n_jobs=1, nthread=None, objective='reg:linear', random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
             silent=True, subsample=0.75)

xgboost.plot_importance(xgb_model)
fig, ax = plt.subplots(figsize=(10,12))
plot_importance(xgb_model, ax=ax)
predictions=xgb_model.predict(df4_x_test)
print(predictions)


plt.show()

from tensorflow.python.keras.metrics import accuracy


r_sq=xgb_model.score(df4_x_train, df4_y_train)
print(r_sq)
print(explained_variance_score(predictions, df4_y_test))

# test_loss, test_acc = xgb_model.evaluate(df4_x_test, df4_y_test)
# print(" accuracy :%.2f%%" % (test_acc*100.0))
'''


'''
r_sq : 0.9998540178355485
explained_variance_score : 0.9984578066997549
'''