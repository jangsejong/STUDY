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
import matplotlib.pyplot as plt

count_data = datasets.groupby('quality')['quality'].count()
plt.bar(count_data.index, count_data)
plt.show()

# p = pd.DataFrame({'count' : datasets.groupby( [ "quality"] ).size()})
# print(p)
# p.plot(kind='bar', rot=0)

# import matplotlib.pyplot as plt
# datasets.groupby( [ "quality"] ).count().plot(kind='bar', rot=0)
# plt.show()

# g1 = datasets.groupby( [ "quality"] ).count()
# g1.plot(kind='bar', rot=0)
# plt.show()



# x = datasets.drop('quality', axis=1)
# x = x.astype(int)
# y = datasets['quality']

# df = datasets
# main_category =x
# sub_category =y
# # print(datasets.isnull().sum())

# import matplotlib.pyplot as plt
# def draw_group_barchart(df,main_category,sub_category,fig_width=10,fig_height=10, \
#                         bar_type='vertical', between_bar_padding=0.85,\
#                         within_bar_padding=0.8, config_bar=None):
#     '''
#     Description :
#     그룹바 차트를 그려주는 함수다. 
    
#     Arguments :
#     df = 메인 카테고리와 서브 카테고리로 이루어진 데이터, pd.DataFrame 객체여야 한다.
#     main_category = 메인 카테고리 변수를 나타내는 문자열
#     sub_category = 서브 카테고리 변수를 모아 놓은 리스트
#     fig_width = 캔버스 폭
#     fig_height = 캔버스 높이
#     bar_type = 'vertical' 또는 'horizontal'값을 가질 수 있으며
#                'vertical'은 수직 바 차트를 'horizontal'은 수평 바 차트를 그린다.
#     between_bar_padding = 메인 카테고리 간 여백 조절 0~1사이의 값을 갖는다.
#     within_bar_padding = 메인 카테고리 내 여백 조절 0~1사이의 값을 갖는다. 
#     config_bar = 바 차트를 꾸미기 위한 옵션. 딕셔너리 형태로 넣어줘야 한다.
    
#     Return : 
#     그룹바 차트 출력
#     '''
    
#     ## Arguments 체크
#     if not isinstance(main_category,str):
#         print(f'main_category인자의 타입은 {type(main_category)}가 아니고 문자열 입니다.')
#         return
#     if not main_category in df.columns:
#         print(f'데이터가 {main_category} 칼럼을 포함하고 있지 않습니다.')
#         return
#     if not set(sub_category).issubset(set(df.columns)):
#         print(f'{set(sub_category)-set(df.columns)}가 데이터에 없습니다.')
#         return
#     if isinstance(bar_type,str):
#         if not bar_type in ['vertical','horizontal']:
#             print(f'bar_type인자에는 "vertical"과 "horizontal"만 허용됩니다.')
#             return
#     else:
#         print(f'bar_type인자의 타입은 {type(bar_type)}가 아니고 문자열 입니다.')
#         return
    
#     if between_bar_padding < 0 or between_bar_padding > 1:
#         print(f'between_bar_padding은 0보다 크거나 같고, 1보다 작거나 같아야합니다.')
#         return
#     if within_bar_padding < 0 or within_bar_padding > 1:
#         print(f'within_bar_padding은 0보다 크거나 같고, 1보다 작거나 같아야합니다.')
#         return
    
#     ## 필요 모듈 임포트
#     import matplotlib.pyplot as plt
#     import seaborn as sns
#     import numpy as np
#     # %matplotlib inline 
 
#     num_sub_category = len(sub_category) ## 서브 카테고리 개수
 
#     fig = plt.figure(figsize=(fig_width,fig_height)) ## 캔버스 생성
#     fig.set_facecolor('white') ## 캔버스 색상 지정
#     ax = fig.add_subplot() ## 그림이 그려질 축을 생성
    
#     colors = sns.color_palette('hls',num_sub_category) ## 막대기 색상 지정
    
#     tick_label = list(df[main_category].unique()) ## 메인 카테고리 라벨 생성
#     tick_number = len(tick_label) ## 메인 카테고리 눈금 개수
    
#     tick_coord = np.arange(tick_number) ## 메인 카테고리안에서 첫번째 서브 카테고리 막대기가 그려지는 x좌표
 
#     width = 1/num_sub_category*between_bar_padding ## 막대기 폭 지정
 
#     config_tick = dict()
#     config_tick['ticks'] = [t + width*(num_sub_category-1)/2 for t in tick_coord] ## 메인 카테고리 라벨 x좌표
#     config_tick['labels'] = tick_label 
 
#     if bar_type == 'vertical': ## 수직 바 차트를 그린다.
#         plt.xticks(**config_tick) ## x축 눈금 라벨 생성
 
#         for i in range(num_sub_category):
#             if config_bar: ## 바 차트 추가 옵션이 있는 경우
#                 ax.bar(tick_coord+width*i, df[sub_category[i]], \
#                        width*within_bar_padding, label=sub_category[i], \
#                        color=colors[i], **config_bar) ## 수직 바 차트 생성
#             else:
#                 ax.bar(tick_coord+width*i, df[sub_category[i]], \
#                        width*within_bar_padding, label=sub_category[i], \
#                        color=colors[i]) ## 수직 바 차트 생성
#         plt.legend() ## 범례 생성
#         plt.savefig('fig03.png',format='png',dpi=300)
#         plt.show()
#     else: ## 수평 바 차트를 그린다.
#         plt.yticks(**config_tick) ## x축 눈금 라벨 생성
 
#         for i in range(num_sub_category):
#             if config_bar: # 바 차트 추가 옵션이 있는 경우
#                 ax.barh(tick_coord+width*i, df[sub_category[i]], \
#                        width*within_bar_padding, label=sub_category[i], \
#                         color=colors[i], **config_bar) ## 수평 바 차트 생성
#             else:
#                 ax.barh(tick_coord+width*i, df[sub_category[i]], \
#                        width*within_bar_padding, label=sub_category[i], \
#                        color=colors[i]) ## 수평 바 차트 생성
#         plt.legend() ## 범례 생성
#         plt.show()

# draw_group_barchart(df,main_category,sub_category)
# plt.show()

