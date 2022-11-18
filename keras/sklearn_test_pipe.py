import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import preprocessing
import seaborn as sns

#데이터 로드

path = 'D:\\Study\\개인프로젝트\\데이터자료\\csv\\'

covid_19 = pd.read_csv(path +"코로나바이러스감염증-19_확진환자_발생현황_220106.csv", thousands=",", encoding='cp949')
kospi = pd.read_csv(path +"코스피지수(202001~202112).csv", thousands=",", encoding='cp949')

# print(covid_19.head())
# print(kospi.head())


#데이터 전처리

#covid_19.csv 에서 결측치 칼럼이 확인되어 삭제해주었다.

# covid_19 = covid_19.drop(covid_19.index[range(0,1)] ,axis=0)
covid_19 = covid_19.drop(covid_19.columns[range(5,8)], axis=1)
covid_19 = covid_19.replace("-","0")


#covid_19 데이터와 상관 관계를 위하여 날짜를 맞추기 위해 kospi 데이터 1월19일까지의 데이터를 삭제 해주었다.
kospi = kospi.drop(kospi.index[range(0,12)] ,axis=0)

# print(covid_19.head())
# print(kospi.info())


# 기존 일자를 new_data 로 수정후 일자 삭제, new_data를 인덱스로 넣어 주었다.
covid_19['new_Date'] = pd.to_datetime(covid_19['일자'])
kospi['new_Date'] = pd.to_datetime(kospi['일자'])

covid_19.drop('일자', axis = 1, inplace=True)
covid_19.set_index('new_Date', inplace=True)

kospi.drop('일자', axis = 1, inplace=True)
kospi.set_index('new_Date', inplace=True)
# print(x1.head())
# print('\n')
# print(x1.info())
# print('\n')
# print(type(x1['new_Date'][1]))

            
x1 = covid_19.drop(columns=[], axis=1) 
x2 = kospi.drop(columns =['대비','등락률(%)', '배당수익률(%)', '주가이익비율', '주가자산비율', '거래량(천주)', '상장시가총액(백만원)', '거래대금(백만원)'], axis=1) 

# for col1 in x1.columns:
#     n_nan1 = x1[col1].isnull().sum()
#     if n_nan1>0:
#       msg1 = '{:^20}에서 결측치 개수 : {}개'.format(col1,n_nan1)
#       print(msg1)
#     else:
        #   print('결측치가 없습니다.')    
# for col2 in x2.columns:
#     n_nan2 = x2[col2].isnull().sum()
#     if n_nan2>0:
#       msg2 = '{:^20}에서 결측치 개수 : {}개'.format(col2,n_nan2)
#       print(msg2)
#     else:
#         #   print('결측치가 없습니다.')  

# print(x1.head())
# print(x2.head())


#covid_19.csv 에서 주말, 공휴일 등 주식장이 열리지 않는 행을 삭제해 주었다.

# delete_sat = pd.date_range('2020-01-20', '2022-01-02', freq='W-SAT')
# print(delete_sat)

#주식장이 열리지 않는 주말에 해당하는 행을 삭제하여 준다.
x1 = x1.drop(pd.date_range('2020-01-20', '2022-01-02', freq='W-SAT'),axis=0)
x1 = x1.drop(pd.date_range('2020-01-20', '2022-01-02', freq='W-SUN'),axis=0)


#주식장이 열리지 않는 공휴일,임시공휴일에 해당하는 행을 삭제하여 준다.
import holidays
kr_holidays = holidays.KR(years=[2020,2021,2022])
# print(kr_holidays)

'''
{datetime.date(2020, 1, 1): "New Year's Day", datetime.date(2020, 1, 24): "The day preceding of Lunar New Year's Day", datetime.date(2020, 1, 25): "Lunar New Year's Day", datetime.date(2020, 1, 26): 
"The second day of Lunar New Year's Day", datetime.date(2020, 1, 27): "Alternative holiday of Lunar New Year's Day", datetime.date(2020, 3, 1): 'Independence Movement Day', 
datetime.date(2020, 4, 30): 'Birthday of the Buddha', datetime.date(2020, 5, 5): "Children's Day", datetime.date(2020, 5, 1): 'Labour Day', datetime.date(2020, 6, 6): 'Memorial Day', 
datetime.date(2020, 8, 15): 'Liberation Day', datetime.date(2020, 9, 30): 'The day preceding of Chuseok', datetime.date(2020, 10, 1): 'Chuseok', datetime.date(2020, 10, 2): 'The second day of Chuseok', 
datetime.date(2020, 10, 3): 'National Foundation Day', datetime.date(2020, 10, 9): 'Hangeul Day', datetime.date(2020, 12, 25): 'Christmas Day', datetime.date(2020, 8, 17): 'Alternative public holiday'}
{datetime.date(2021, 1, 1): "New Year's Day", datetime.date(2021, 2, 11): "The day preceding of Lunar New Year's Day", datetime.date(2021, 2, 12): "Lunar New Year's Day", datetime.date(2021, 2, 13): 
"The second day of Lunar New Year's Day", datetime.date(2021, 3, 1): 'Independence Movement Day', datetime.date(2021, 5, 19): 'Birthday of the Buddha', datetime.date(2021, 5, 5): "Children's Day", 
datetime.date(2021, 5, 1): 'Labour Day', datetime.date(2021, 6, 6): 'Memorial Day', datetime.date(2021, 8, 15): 'Liberation Day', datetime.date(2021, 8, 16): 'Alternative holiday of Liberation Day', 
datetime.date(2021, 9, 20): 'The day preceding of Chuseok', datetime.date(2021, 9, 21): 'Chuseok', datetime.date(2021, 9, 22): 'The second day of Chuseok', datetime.date(2021, 10, 3): 'National Foundation Day', 
datetime.date(2021, 10, 4): 'Alternative holiday of National Foundation Day', datetime.date(2021, 10, 9): 'Hangeul Day', datetime.date(2021, 10, 11): 'Alternative holiday of Hangeul Day', datetime.date(2021, 12, 25): 'Christmas Day'}
'''
x1.drop("2020-01-24", axis=0, inplace=True)
# x1.drop("2020-01-25", axis=0, inplace=True)
# x1.drop("2020-01-26", axis=0, inplace=True)
x1.drop("2020-01-27", axis=0, inplace=True)
# x1.drop("2020-03-01", axis=0, inplace=True)
x1.drop("2020-04-15", axis=0, inplace=True)  #제21대 국회의원선거
x1.drop("2020-04-30", axis=0, inplace=True)
x1.drop("2020-05-05", axis=0, inplace=True)
x1.drop("2020-05-01", axis=0, inplace=True)
# x1.drop("2020-06-06", axis=0, inplace=True)
# x1.drop("2020-08-15", axis=0, inplace=True)
x1.drop("2020-09-30", axis=0, inplace=True)
x1.drop("2020-10-01", axis=0, inplace=True)
x1.drop("2020-10-02", axis=0, inplace=True)
x1.drop("2020-12-31", axis=0, inplace=True)
x1.drop("2020-10-09", axis=0, inplace=True)
x1.drop("2020-08-17", axis=0, inplace=True)
x1.drop("2020-12-25", axis=0, inplace=True)
x1.drop("2021-01-01", axis=0, inplace=True)
# x1.drop("2020-12-05", axis=0, inplace=True)
# x1.drop("2021-01-03", axis=0, inplace=True)
x1.drop("2021-02-11", axis=0, inplace=True)
x1.drop("2021-02-12", axis=0, inplace=True)
# x1.drop("2021-02-13", axis=0, inplace=True)
x1.drop("2021-03-01", axis=0, inplace=True)
x1.drop("2021-05-05", axis=0, inplace=True)
x1.drop("2021-05-19", axis=0, inplace=True)
# x1.drop("2021-05-01", axis=0, inplace=True)
# x1.drop("2021-06-06", axis=0, inplace=True)
# x1.drop("2021-08-15", axis=0, inplace=True)
x1.drop("2021-08-16", axis=0, inplace=True)
x1.drop("2021-09-20", axis=0, inplace=True)
x1.drop("2021-09-21", axis=0, inplace=True)
x1.drop("2021-09-22", axis=0, inplace=True)
# x1.drop("2021-10-03", axis=0, inplace=True)
x1.drop("2021-10-04", axis=0, inplace=True)
# x1.drop("2021-10-09", axis=0, inplace=True)
x1.drop("2021-10-11", axis=0, inplace=True)
x1.drop("2021-12-31", axis=0, inplace=True)
# x1.drop("2022-01-01", axis=0, inplace=True)




# print(x1.info)
# print(x2.info)
# print(x1.head)
# print(x2.head)
# print(x1.shape)
# print(x2.shape)



# from pytimekr import pytimekr
 
# x1 = np.array(x1)
# x2 = np.array(x2)
# print(x1.head, x2.head)

# x1.plot(x='일자', y='계(명)')
# x2.plot(x='일자', y='시가지수')

# plt.figure(figsize = (9, 5))
# plt.grid()
# plt.legend(loc = 'upper right')
# plt.show()



print(x1.shape, x2.shape) #(484, 4) (484, 4)


x11 = np.array(x1)
x22 = np.array(x2)


# train_data = pd.read_csv(path+'train.csv').drop(['id'],axis=1)
# test_data = pd.read_csv(path+'test.csv').drop(['id'],axis=1)

# train_data['Sex']=np.where(train_data['Sex'].values=='MALE',1,np.where(train_data['Sex'].values=='FEMALE',0,np.nan))
# test_data['Sex']=np.where(test_data['Sex'].values=='MALE',1,np.where(test_data['Sex'].values=='FEMALE',0,np.nan))
# train_data['Clutch Completion']=np.where(train_data['Clutch Completion'].values=='Yes',1,0)
# test_data['Clutch Completion']=np.where(test_data['Clutch Completion'].values=='Yes',1,0)
# train = pd.concat([train_data,pd.get_dummies(train_data[['Island','Species']])],axis=1)
# test = pd.concat([test_data,pd.get_dummies(test_data[['Island','Species']])],axis=1)
# train = train.drop(['Island','Species','Delta 15 N (o/oo)',  'Delta 13 C (o/oo)'],axis=1)
# test = test.drop(['Island','Species','Delta 15 N (o/oo)',  'Delta 13 C (o/oo)'],axis=1)


# Base

num_folds= 10
seed = 7
scoring = 'neg_root_mean_squared_error'
X_all = train[test.columns.tolist()]
y_all =train['Body Mass (g)']



X_train, X_valid, y_train, y_valid = train_test_split(train[test.columns.tolist()],train['Body Mass (g)']
                                                      ,test_size=0.25,random_state=66)


models = []
models.append(('LR',LinearRegression()))
models.append(('LASSO',Lasso()))
models.append(('KNN',KNeighborsRegressor()))
models.append(('CART',DecisionTreeRegressor()))
models.append(('EN',ElasticNet()))
models.append(('SVM',SVR()))
models.append(('RFR',RandomForestRegressor()))
models.append(('XGBR',XGBRegressor()))
models.append(('LGBMR',LGBMRegressor()))
models.append(('AdaR',AdaBoostRegressor()))
models.append(('Cat',CatBoostRegressor(verbose=False)))
models.append(('Xtree',ExtraTreesRegressor()))

results =[]
names = []
for name, model in models:
  kfold = KFold(n_splits=10,random_state=66,shuffle = True)
  cv_results = cross_val_score(model,X_train,y_train
                               ,cv= kfold,scoring=scoring)
  results.append(cv_results)
  names.append(name)
  msg = "%s : %f (%f) "%(name,cv_results.mean(),cv_results.std())
  print(msg)




#standardization

pipelines = []
pipelines.append(('ScaledLR',Pipeline([('Scaler',preprocessing.StandardScaler()),('LR',LinearRegression())])))
pipelines.append(('ScaledLASSO',Pipeline([('Scaler',preprocessing.StandardScaler()),('LASSO',Lasso())])))
pipelines.append(('ScaledKNN',Pipeline([('Scaler',preprocessing.StandardScaler()),('KNN',KNeighborsRegressor())])))
pipelines.append(('ScaledCART',Pipeline([('Scaler',preprocessing.StandardScaler()),('CART',DecisionTreeRegressor())])))
pipelines.append(('ScaledEN',Pipeline([('Scaler',preprocessing.StandardScaler()),('EN',ElasticNet())])))
pipelines.append(('ScaledSVM',Pipeline([('Scaler',preprocessing.StandardScaler()),('SVM',SVR())])))
pipelines.append(('ScaledRFR',Pipeline([('Scaler',preprocessing.StandardScaler()),('RFR',RandomForestRegressor())])))
pipelines.append(('ScaledXGBR',Pipeline([('Scaler',preprocessing.StandardScaler()),('XGBR',XGBRegressor())])))
pipelines.append(('ScaledLGBMR',Pipeline([('Scaler',preprocessing.StandardScaler()),('LGBMR',LGBMRegressor())])))
pipelines.append(('ScaledAdaR',Pipeline([('Scaler',preprocessing.StandardScaler()),('AdaR',AdaBoostRegressor())])))
pipelines.append(('ScaledCat',Pipeline([('Scaler',preprocessing.StandardScaler()),('Cat',CatBoostRegressor(verbose=False))])))
pipelines.append(('ScaledXtree',Pipeline([('Scaler',preprocessing.StandardScaler()),('Xtree',ExtraTreesRegressor())])))

results_scaled =[]
names_scaled = []
for name, model in pipelines:
  kfold = KFold(n_splits=10,random_state=66,shuffle = True)
  cv_results = cross_val_score(model,X_train,y_train
                               ,cv= kfold,scoring=scoring)
  results_scaled.append(cv_results)
  names_scaled.append(name)
  msg = "%s : %f (%f) "%(name,cv_results.mean(),cv_results.std())
  print(msg)
  
  
  
  
  
  
  
  
  
  
scaler = preprocessing.StandardScaler().fit(X_all)
scaled_X = scaler.transform(X_all)
params = { 'n_estimators' : [10, 50,100],
           'max_depth' : [6, 12,18,24],
           'min_samples_leaf' : [1, 6, 12, 18],
           'min_samples_split' : [2, 8, 16, 20]
            }
model = RandomForestRegressor()
kfold = KFold(n_splits= num_folds,random_state = 66 ,shuffle = True)
grid = GridSearchCV(estimator= model, param_grid = params,scoring= 'neg_root_mean_squared_error',cv=kfold )
grid_result = grid.fit(scaled_X,y_all)

print("Best : %f using %s "%(grid_result.best_score_,grid_result.best_params_))  
  
  
  
  
params = { 'n_estimators' : [10, 50,100],
           'max_depth' : [6,12,18,24],
           'min_samples_leaf' : [1, 6, 12, 18],
           'min_samples_split' : [2,4,8, 16]
            }
model =ExtraTreesRegressor()
kfold = KFold(n_splits= num_folds,random_state = 66 ,shuffle = True)
grid = GridSearchCV(estimator= model, param_grid = params, scoring= 'neg_root_mean_squared_error',cv=kfold )
grid_result = grid.fit(X_all,y_all)  
  
  

print("Best : %f using %s "%(grid_result.best_score_,grid_result.best_params_))

  
  
from sklearn.metrics import mean_squared_error
import math   
  
  
errors = []
pred_valid=[]
pred_test = []  
  
  
scaler = preprocessing.StandardScaler().fit(X_train)
scaled_X_train = scaler.transform(X_train)
scaled_X_valid = scaler.transform(X_valid)
scaled_X_test = scaler.transform(test) 
  
  
  
lasso = Lasso()
lasso.fit(X_train,y_train)
lasso_valid = lasso.predict(X_valid)
rmse = math.sqrt(mean_squared_error(y_valid, lasso_valid))
errors.append(('Lasso',rmse))
pred_valid.append(('Lasso',lasso_valid))
lasso_test = lasso.predict(test)
pred_test.append(('Lasso',lasso_test))  
  
  
LR =LinearRegression()
LR.fit(scaled_X_train,y_train)
lr_valid = LR.predict(scaled_X_valid)
rmse = math.sqrt(mean_squared_error(y_valid, lr_valid))
errors.append(('LR',rmse))
pred_valid.append(('LR',lr_valid))
lr_test = LR.predict(scaled_X_test)
pred_test.append(('LR',lr_test))  
  
  
RF =RandomForestRegressor(max_depth= 24, min_samples_leaf= 12, min_samples_split= 16, n_estimators= 100)
RF.fit(scaled_X_train,y_train)
rf_valid = RF.predict(scaled_X_valid)
rmse = math.sqrt(mean_squared_error(y_valid, rf_valid))
errors.append(('RF',rmse))
pred_valid.append(('RF',rf_valid))
rf_test = RF.predict(scaled_X_test)
pred_test.append(('RF',rf_test))  
  
ET =ExtraTreesRegressor(max_depth=24, min_samples_leaf= 1, min_samples_split= 8, n_estimators= 100)
ET.fit(X_train,y_train)
et_valid = ET.predict(X_valid)
rmse = math.sqrt(mean_squared_error(y_valid, et_valid))
errors.append(('ET',rmse))
pred_valid.append(('ET',et_valid))
et_test = ET.predict(test)
pred_test.append(('ET',et_test))  
  
CAT = CatBoostRegressor(iterations=10000,random_state=66
           ,eval_metric="RMSE")
CAT.fit(X_train,y_train, eval_set=[(X_valid,y_valid)],early_stopping_rounds=19
        ,verbose=1000 )
cat_valid = CAT.predict(X_valid)
rmse = math.sqrt(mean_squared_error(y_valid, cat_valid))
errors.append(('CAT',rmse))
pred_valid.append(('CAT',cat_valid))
cat_test = CAT.predict(test)
pred_test.append(('CAT',cat_test))


  
  
for name, error in errors:
      print("{} : {}".format(name,error)) 
  
  
val= np.zeros(X_valid.shape[0])
for name, pred in pred_valid:
  val+= (0.2* pred)
math.sqrt(mean_squared_error(y_valid, val))  
  
  
  
  
val= np.zeros(X_valid.shape[0])
for name, pred in pred_valid:
  if name == 'Lasso' or name=='LR' or name == 'ET' or name=='CAT':
    val+= (0.25* pred)
math.sqrt(mean_squared_error(y_valid, val)) 
  
  
  
test_val= np.zeros(test.shape[0])
for name, pred in pred_test:
  if name == 'Lasso' or name=='LR' or name == 'ET' or name=='CAT':
    test_val+= (0.25* pred)

#model.save_weights("./_save/keras999_1_save_weights.h5")
#model = load_model('./_ModelCheckPoint/ss_ki_1222_Trafevol5.hdf5')
  
submission = pd.read_csv(path+'sample_submission.csv')
submission['Body Mass (g)'] = test_val
submission.to_csv(path+"penguin_0106_01.csv", index=False)
