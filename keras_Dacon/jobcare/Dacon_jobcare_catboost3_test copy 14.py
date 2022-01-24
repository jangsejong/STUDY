


DATA_PATH = 'D:\\Study\\_data\\dacon\\Jobcare\\'
SUBMIT_PATH = 'D:\\Study\\_data\\dacon\\Jobcare\\'
SEED = 66

import os
import sys
import platform
import random
import math
from typing import List ,Dict, Tuple

import pandas as pd
import numpy as np
 
import sklearn 
from sklearn.model_selection import StratifiedKFold , KFold
from sklearn.metrics import f1_score 

from catboost import Pool,CatBoostClassifier

print(f"- os: {platform.platform()}")
print(f"- python: {sys.version}")
print(f"- pandas: {pd.__version__}")
print(f"- numpy: {np.__version__}")
print(f"- sklearn: {sklearn.__version__}")


train_data = pd.read_csv(f'{DATA_PATH}train.csv')
test_data = pd.read_csv(f'{DATA_PATH}test.csv')
sample_submission = pd.read_csv(f'{DATA_PATH}sample_submission.csv')

code_d = pd.read_csv(f'{DATA_PATH}속성_D_코드.csv').iloc[:,:-1]
code_h = pd.read_csv(f'{DATA_PATH}속성_H_코드.csv')
code_l = pd.read_csv(f'{DATA_PATH}속성_L_코드.csv')

train_data.shape , test_data.shape

code_d.columns= ["attribute_d_d","attribute_d_s","attribute_d_m","attribute_d_l"]
code_h.columns= ["attribute_","attribute_h","attribute_h_p"]
code_l.columns= ["attribute_l","attribute_l_d","attribute_l_s","attribute_l_m","attribute_l_l",]


def merge_codes(df:pd.DataFrame,df_code:pd.DataFrame,col:str)->pd.DataFrame:
    df = df.copy()
    df_code = df_code.copy()
    df_code = df_code.add_prefix(f"{col}_")
    df_code.columns.values[0] = col
    return pd.merge(df,df_code,how="left",on=col)


def preprocess_data(
                    df:pd.DataFrame,is_train:bool = True, cols_merge:List[Tuple[str,pd.DataFrame]] = []  , cols_equi:List[Tuple[str,str]]= [] ,
                    cols_drop:List[str] = ["id","person_prefer_f","person_prefer_g" ,"contents_open_dt"]
                    )->Tuple[pd.DataFrame,np.ndarray]:
    df = df.copy()

    y_data = None
    if is_train:
        y_data = df["target"].to_numpy()
        df = df.drop(columns="target")

    for col, df_code in cols_merge:
        df = merge_codes(df,df_code,col)

    cols = df.select_dtypes(bool).columns.tolist()
    df[cols] = df[cols].astype(int)

    for col1, col2 in cols_equi:
        df[f"{col1}_{col2}"] = (df[col1] == df[col2] ).astype(int)

    df = df.drop(columns=cols_drop)
    return (df , y_data)



# 소분류 중분류 대분류 속성코드 merge 컬럼명 및 데이터 프레임 리스트
cols_merge = [
              ("person_prefer_d_1" , code_d),
              ("person_prefer_d_2" , code_d),
              ("person_prefer_d_3" , code_d),
              ("contents_attribute_d" , code_d),
              ("person_prefer_h_1" , code_h),
              ("person_prefer_h_2" , code_h),
              ("person_prefer_h_3" , code_h),
              ("contents_attribute_h" , code_h),
              ("contents_attribute_l" , code_l),
]

# 회원 속성과 콘텐츠 속성의 동일한 코드 여부에 대한 컬럼명 리스트
cols_equi = [

    ("contents_attribute_c","person_prefer_c"),
    ("contents_attribute_e","person_prefer_e"),

    ("person_prefer_d_2_attribute_d_s" , "contents_attribute_d_attribute_d_s"),
    ("person_prefer_d_2_attribute_d_m" , "contents_attribute_d_attribute_d_m"),
    ("person_prefer_d_2_attribute_d_l" , "contents_attribute_d_attribute_d_l"),
    ("person_prefer_d_3_attribute_d_s" , "contents_attribute_d_attribute_d_s"),
    ("person_prefer_d_3_attribute_d_m" , "contents_attribute_d_attribute_d_m"),
    ("person_prefer_d_3_attribute_d_l" , "contents_attribute_d_attribute_d_l"),

    ("person_prefer_h_1_attribute_h_p" , "contents_attribute_h_attribute_h_p"),
    ("person_prefer_h_2_attribute_h_p" , "contents_attribute_h_attribute_h_p"),
    ("person_prefer_h_3_attribute_h_p" , "contents_attribute_h_attribute_h_p"),

]

# 학습에 필요없는 컬럼 리스트
cols_drop = ["id","person_prefer_f","person_prefer_g" ,"contents_open_dt", "contents_rn", "person_rn",]





x_train, y_train = preprocess_data(train_data, cols_merge = cols_merge , cols_equi= cols_equi , cols_drop = cols_drop)
x_test, _ = preprocess_data(test_data,is_train = False, cols_merge = cols_merge , cols_equi= cols_equi  , cols_drop = cols_drop)
x_train.shape , y_train.shape , x_test.shape


cat_features = x_train.columns[x_train.nunique() > 2].tolist()


is_holdout = False
n_splits = 5
iterations = 20000
patience = 2000

cv = KFold(n_splits=n_splits, shuffle=True, random_state=66)

scores = []
models = []


models = []
for tri, vai in cv.split(x_train):
    print("="*50)
    preds = []

    model = CatBoostClassifier(iterations=iterations,random_state=66,task_type="GPU",eval_metric="F1",cat_features=cat_features,one_hot_max_size=4)
    model.fit(x_train.iloc[tri], y_train[tri], 
            eval_set=[(x_train.iloc[vai], y_train[vai])], 
            early_stopping_rounds=patience ,
            verbose = 200
        )
    
    models.append(model)
    scores.append(model.get_best_score()["validation"]["F1"])
    if is_holdout:
        break  


print(scores)
print(np.mean(scores))




threshold = 0.35

pred_list = []
scores = []
for i,(tri, vai) in enumerate( cv.split(x_train) ):
    pred = models[i].predict_proba(x_train.iloc[vai])[:, 1]
    pred = np.where(pred >= threshold , 1, 0)
    score = f1_score(y_train[vai],pred)
    scores.append(score)
    pred = models[i].predict_proba(x_test)[:, 1]
    pred_list.append(pred)
print(scores)
print(np.mean(scores))

pred = np.mean( pred_list , axis = 0 )
pred = np.where(pred >= threshold , 1, 0)

# sample_submission = pd.load_csv(f'{DATA_PATH}sample_submission.csv')
sample_submission['target'] = pred
sample_submission
sample_submission.to_csv(DATA_PATH + "jobcare_0118_7_10_7.csv", index=False)  



'''
jobcare_0118_7_10_5
- os: Windows-10-10.0.19042-SP0
- python: 3.9.7 (default, Sep 16 2021, 16:59:28) [MSC v.1916 64 bit (AMD64)]
- pandas: 1.3.4
- numpy: 1.19.5
- sklearn: 0.24.2
==================================================
Learning rate set to 0.013941
0:      learn: 0.6094207        test: 0.6131333 best: 0.6131333 (0)     total: 151ms    remaining: 37m 39s
200:    learn: 0.6409863        test: 0.6521999 best: 0.6521999 (200)   total: 26.4s    remaining: 32m 22s
400:    learn: 0.6521311        test: 0.6686936 best: 0.6687914 (398)   total: 50.9s    remaining: 30m 53s
600:    learn: 0.6577744        test: 0.6725621 best: 0.6725783 (586)   total: 1m 14s   remaining: 29m 50s
800:    learn: 0.6615398        test: 0.6753472 best: 0.6753716 (799)   total: 1m 37s   remaining: 28m 53s
1000:   learn: 0.6647007        test: 0.6783187 best: 0.6785661 (977)   total: 2m 1s    remaining: 28m 15s
1200:   learn: 0.6669904        test: 0.6796601 best: 0.6796831 (1197)  total: 2m 24s   remaining: 27m 37s
1400:   learn: 0.6688083        test: 0.6797701 best: 0.6802402 (1368)  total: 2m 46s   remaining: 26m 59s
1600:   learn: 0.6706251        test: 0.6800357 best: 0.6802402 (1368)  total: 3m 9s    remaining: 26m 24s
1800:   learn: 0.6717956        test: 0.6800481 best: 0.6804033 (1757)  total: 3m 31s   remaining: 25m 49s
2000:   learn: 0.6729439        test: 0.6799606 best: 0.6804033 (1757)  total: 3m 53s   remaining: 25m 17s
2200:   learn: 0.6739186        test: 0.6795649 best: 0.6804033 (1757)  total: 4m 15s   remaining: 24m 45s
bestTest = 0.6804033402
bestIteration = 1757
Shrink model to first 1758 iterations.
==================================================
Learning rate set to 0.013941
0:      learn: 0.6355245        test: 0.6312463 best: 0.6312463 (0)     total: 149ms    remaining: 37m 7s
200:    learn: 0.6412832        test: 0.6485143 best: 0.6485143 (200)   total: 26.9s    remaining: 33m 2s
400:    learn: 0.6519832        test: 0.6648176 best: 0.6649041 (398)   total: 51.5s    remaining: 31m 16s
600:    learn: 0.6570702        test: 0.6704354 best: 0.6704477 (599)   total: 1m 15s   remaining: 30m 10s
800:    learn: 0.6611016        test: 0.6737337 best: 0.6737817 (794)   total: 1m 38s   remaining: 29m 11s
1000:   learn: 0.6643910        test: 0.6754659 best: 0.6755042 (999)   total: 2m 2s    remaining: 28m 26s
1200:   learn: 0.6668251        test: 0.6765828 best: 0.6766381 (1166)  total: 2m 24s   remaining: 27m 44s
1400:   learn: 0.6690453        test: 0.6779212 best: 0.6779464 (1399)  total: 2m 47s   remaining: 27m 5s
1600:   learn: 0.6709513        test: 0.6784675 best: 0.6785195 (1591)  total: 3m 9s    remaining: 26m 29s
1800:   learn: 0.6722921        test: 0.6789990 best: 0.6791360 (1722)  total: 3m 32s   remaining: 25m 55s
2000:   learn: 0.6736196        test: 0.6790415 best: 0.6791561 (1821)  total: 3m 54s   remaining: 25m 23s
2200:   learn: 0.6748304        test: 0.6783013 best: 0.6791561 (1821)  total: 4m 16s   remaining: 24m 53s
bestTest = 0.6791561441
bestIteration = 1821
Shrink model to first 1822 iterations.
==================================================
Learning rate set to 0.013941
0:      learn: 0.6193509        test: 0.6224305 best: 0.6224305 (0)     total: 148ms    remaining: 37m 5s
200:    learn: 0.6402938        test: 0.6525579 best: 0.6525579 (200)   total: 26.4s    remaining: 32m 22s
400:    learn: 0.6507617        test: 0.6658023 best: 0.6658892 (390)   total: 51.2s    remaining: 31m 4s
600:    learn: 0.6568302        test: 0.6725165 best: 0.6725911 (599)   total: 1m 14s   remaining: 29m 53s
800:    learn: 0.6610897        test: 0.6759036 best: 0.6759094 (799)   total: 1m 38s   remaining: 28m 57s
1000:   learn: 0.6645726        test: 0.6785962 best: 0.6786540 (986)   total: 2m 1s    remaining: 28m 12s
1200:   learn: 0.6669423        test: 0.6796691 best: 0.6796748 (1197)  total: 2m 23s   remaining: 27m 32s
1400:   learn: 0.6688402        test: 0.6801504 best: 0.6802969 (1376)  total: 2m 46s   remaining: 26m 54s
1600:   learn: 0.6704840        test: 0.6798829 best: 0.6802988 (1479)  total: 3m 8s    remaining: 26m 20s
1800:   learn: 0.6717454        test: 0.6796569 best: 0.6802988 (1479)  total: 3m 31s   remaining: 25m 46s
bestTest = 0.6802987703
bestIteration = 1479
Shrink model to first 1480 iterations.
==================================================
Learning rate set to 0.013941
0:      learn: 0.6091699        test: 0.6117842 best: 0.6117842 (0)     total: 147ms    remaining: 36m 43s
200:    learn: 0.6416134        test: 0.6527523 best: 0.6527681 (199)   total: 26.2s    remaining: 32m 5s
400:    learn: 0.6518610        test: 0.6669491 best: 0.6669859 (399)   total: 50.8s    remaining: 30m 48s
600:    learn: 0.6580400        test: 0.6739970 best: 0.6739970 (600)   total: 1m 14s   remaining: 29m 38s
800:    learn: 0.6619029        test: 0.6777642 best: 0.6777642 (800)   total: 1m 37s   remaining: 28m 47s
1000:   learn: 0.6652794        test: 0.6801662 best: 0.6802896 (996)   total: 2m       remaining: 28m 3s
1200:   learn: 0.6676264        test: 0.6811866 best: 0.6816261 (1161)  total: 2m 23s   remaining: 27m 23s
1400:   learn: 0.6695630        test: 0.6820575 best: 0.6822722 (1391)  total: 2m 45s   remaining: 26m 46s
1600:   learn: 0.6713608        test: 0.6825557 best: 0.6825557 (1600)  total: 3m 7s    remaining: 26m 10s
1800:   learn: 0.6722833        test: 0.6824675 best: 0.6827683 (1734)  total: 3m 29s   remaining: 25m 38s
2000:   learn: 0.6732059        test: 0.6824409 best: 0.6827683 (1734)  total: 3m 52s   remaining: 25m 10s
2200:   learn: 0.6741160        test: 0.6819654 best: 0.6827683 (1734)  total: 4m 15s   remaining: 24m 43s
bestTest = 0.6827683103
bestIteration = 1734
Shrink model to first 1735 iterations.
==================================================
Learning rate set to 0.013941
0:      learn: 0.6243277        test: 0.6210133 best: 0.6210133 (0)     total: 150ms    remaining: 37m 30s
200:    learn: 0.6412986        test: 0.6508174 best: 0.6508174 (200)   total: 26.3s    remaining: 32m 19s
400:    learn: 0.6514146        test: 0.6667217 best: 0.6667461 (398)   total: 50.8s    remaining: 30m 49s
600:    learn: 0.6573675        test: 0.6731169 best: 0.6732531 (595)   total: 1m 14s   remaining: 29m 48s
800:    learn: 0.6615689        test: 0.6765565 best: 0.6766558 (798)   total: 1m 38s   remaining: 29m 4s
1000:   learn: 0.6648934        test: 0.6782555 best: 0.6783360 (999)   total: 2m 2s    remaining: 28m 26s
1200:   learn: 0.6675551        test: 0.6785520 best: 0.6787965 (1162)  total: 2m 24s   remaining: 27m 45s
1400:   learn: 0.6695305        test: 0.6793179 best: 0.6793525 (1328)  total: 2m 47s   remaining: 27m 7s
1600:   learn: 0.6712111        test: 0.6798494 best: 0.6798991 (1574)  total: 3m 10s   remaining: 26m 31s
1800:   learn: 0.6726767        test: 0.6797910 best: 0.6800546 (1611)  total: 3m 32s   remaining: 25m 56s
2000:   learn: 0.6738319        test: 0.6790730 best: 0.6800546 (1611)  total: 3m 54s   remaining: 25m 24s
bestTest = 0.6800546396
bestIteration = 1611
Shrink model to first 1612 iterations.
[0.6804033402207136, 0.6791561440738999, 0.6802987702706541, 0.6827683102831964, 0.6800546396251115]
0.6805362408947151
[0.7103858706085622, 0.7097606177606178, 0.7087576064522517, 0.7099581770428166, 0.7081182902736726]
0.7093961124275843


is_holdout = False
n_splits = 5
iterations = 15000
patience = 1000
jobcare_0118_7_10_6
- os: Windows-10-10.0.19042-SP0
- python: 3.9.7 (default, Sep 16 2021, 16:59:28) [MSC v.1916 64 bit (AMD64)]
- pandas: 1.3.4
- numpy: 1.19.5
- sklearn: 0.24.2
==================================================
Learning rate set to 0.013941
0:      learn: 0.6094207        test: 0.6131333 best: 0.6131333 (0)     total: 151ms    remaining: 37m 39s
200:    learn: 0.6409863        test: 0.6521999 best: 0.6521999 (200)   total: 26.4s    remaining: 32m 22s
400:    learn: 0.6521311        test: 0.6686936 best: 0.6687914 (398)   total: 50.9s    remaining: 30m 53s
600:    learn: 0.6577744        test: 0.6725621 best: 0.6725783 (586)   total: 1m 14s   remaining: 29m 50s
800:    learn: 0.6615398        test: 0.6753472 best: 0.6753716 (799)   total: 1m 37s   remaining: 28m 53s
1000:   learn: 0.6647007        test: 0.6783187 best: 0.6785661 (977)   total: 2m 1s    remaining: 28m 15s
1200:   learn: 0.6669904        test: 0.6796601 best: 0.6796831 (1197)  total: 2m 24s   remaining: 27m 37s
1400:   learn: 0.6688083        test: 0.6797701 best: 0.6802402 (1368)  total: 2m 46s   remaining: 26m 59s
1600:   learn: 0.6706251        test: 0.6800357 best: 0.6802402 (1368)  total: 3m 9s    remaining: 26m 24s
1800:   learn: 0.6717956        test: 0.6800481 best: 0.6804033 (1757)  total: 3m 31s   remaining: 25m 49s
2000:   learn: 0.6729439        test: 0.6799606 best: 0.6804033 (1757)  total: 3m 53s   remaining: 25m 17s
2200:   learn: 0.6739186        test: 0.6795649 best: 0.6804033 (1757)  total: 4m 15s   remaining: 24m 45s
bestTest = 0.6804033402
bestIteration = 1757
Shrink model to first 1758 iterations.
==================================================
Learning rate set to 0.013941
0:      learn: 0.6355245        test: 0.6312463 best: 0.6312463 (0)     total: 149ms    remaining: 37m 7s
200:    learn: 0.6412832        test: 0.6485143 best: 0.6485143 (200)   total: 26.9s    remaining: 33m 2s
400:    learn: 0.6519832        test: 0.6648176 best: 0.6649041 (398)   total: 51.5s    remaining: 31m 16s
600:    learn: 0.6570702        test: 0.6704354 best: 0.6704477 (599)   total: 1m 15s   remaining: 30m 10s
800:    learn: 0.6611016        test: 0.6737337 best: 0.6737817 (794)   total: 1m 38s   remaining: 29m 11s
1000:   learn: 0.6643910        test: 0.6754659 best: 0.6755042 (999)   total: 2m 2s    remaining: 28m 26s
1200:   learn: 0.6668251        test: 0.6765828 best: 0.6766381 (1166)  total: 2m 24s   remaining: 27m 44s
1400:   learn: 0.6690453        test: 0.6779212 best: 0.6779464 (1399)  total: 2m 47s   remaining: 27m 5s
1600:   learn: 0.6709513        test: 0.6784675 best: 0.6785195 (1591)  total: 3m 9s    remaining: 26m 29s
1800:   learn: 0.6722921        test: 0.6789990 best: 0.6791360 (1722)  total: 3m 32s   remaining: 25m 55s
2000:   learn: 0.6736196        test: 0.6790415 best: 0.6791561 (1821)  total: 3m 54s   remaining: 25m 23s
2200:   learn: 0.6748304        test: 0.6783013 best: 0.6791561 (1821)  total: 4m 16s   remaining: 24m 53s
bestTest = 0.6791561441
bestIteration = 1821
Shrink model to first 1822 iterations.
==================================================
Learning rate set to 0.013941
0:      learn: 0.6193509        test: 0.6224305 best: 0.6224305 (0)     total: 148ms    remaining: 37m 5s
200:    learn: 0.6402938        test: 0.6525579 best: 0.6525579 (200)   total: 26.4s    remaining: 32m 22s
400:    learn: 0.6507617        test: 0.6658023 best: 0.6658892 (390)   total: 51.2s    remaining: 31m 4s
600:    learn: 0.6568302        test: 0.6725165 best: 0.6725911 (599)   total: 1m 14s   remaining: 29m 53s
800:    learn: 0.6610897        test: 0.6759036 best: 0.6759094 (799)   total: 1m 38s   remaining: 28m 57s
1000:   learn: 0.6645726        test: 0.6785962 best: 0.6786540 (986)   total: 2m 1s    remaining: 28m 12s
1200:   learn: 0.6669423        test: 0.6796691 best: 0.6796748 (1197)  total: 2m 23s   remaining: 27m 32s
1400:   learn: 0.6688402        test: 0.6801504 best: 0.6802969 (1376)  total: 2m 46s   remaining: 26m 54s
1600:   learn: 0.6704840        test: 0.6798829 best: 0.6802988 (1479)  total: 3m 8s    remaining: 26m 20s
1800:   learn: 0.6717454        test: 0.6796569 best: 0.6802988 (1479)  total: 3m 31s   remaining: 25m 46s
bestTest = 0.6802987703
bestIteration = 1479
Shrink model to first 1480 iterations.
==================================================
Learning rate set to 0.013941
0:      learn: 0.6091699        test: 0.6117842 best: 0.6117842 (0)     total: 147ms    remaining: 36m 43s
200:    learn: 0.6416134        test: 0.6527523 best: 0.6527681 (199)   total: 26.2s    remaining: 32m 5s
400:    learn: 0.6518610        test: 0.6669491 best: 0.6669859 (399)   total: 50.8s    remaining: 30m 48s
600:    learn: 0.6580400        test: 0.6739970 best: 0.6739970 (600)   total: 1m 14s   remaining: 29m 38s
800:    learn: 0.6619029        test: 0.6777642 best: 0.6777642 (800)   total: 1m 37s   remaining: 28m 47s
1000:   learn: 0.6652794        test: 0.6801662 best: 0.6802896 (996)   total: 2m       remaining: 28m 3s
1200:   learn: 0.6676264        test: 0.6811866 best: 0.6816261 (1161)  total: 2m 23s   remaining: 27m 23s
1400:   learn: 0.6695630        test: 0.6820575 best: 0.6822722 (1391)  total: 2m 45s   remaining: 26m 46s
1600:   learn: 0.6713608        test: 0.6825557 best: 0.6825557 (1600)  total: 3m 7s    remaining: 26m 10s
1800:   learn: 0.6722833        test: 0.6824675 best: 0.6827683 (1734)  total: 3m 29s   remaining: 25m 38s
2000:   learn: 0.6732059        test: 0.6824409 best: 0.6827683 (1734)  total: 3m 52s   remaining: 25m 10s
2200:   learn: 0.6741160        test: 0.6819654 best: 0.6827683 (1734)  total: 4m 15s   remaining: 24m 43s
bestTest = 0.6827683103
bestIteration = 1734
Shrink model to first 1735 iterations.
==================================================
Learning rate set to 0.013941
200:    learn: 0.6412986        test: 0.6508174 best: 0.6508174 (200)   total: 26.3s    remaining: 32m 19s
400:    learn: 0.6514146        test: 0.6667217 best: 0.6667461 (398)   total: 50.8s    remaining: 30m 49s
600:    learn: 0.6573675        test: 0.6731169 best: 0.6732531 (595)   total: 1m 14s   remaining: 29m 48s
800:    learn: 0.6615689        test: 0.6765565 best: 0.6766558 (798)   total: 1m 38s   remaining: 29m 4s
1000:   learn: 0.6648934        test: 0.6782555 best: 0.6783360 (999)   total: 2m 2s    remaining: 28m 26s
1200:   learn: 0.6675551        test: 0.6785520 best: 0.6787965 (1162)  total: 2m 24s   remaining: 27m 45s
1400:   learn: 0.6695305        test: 0.6793179 best: 0.6793525 (1328)  total: 2m 47s   remaining: 27m 7s
1600:   learn: 0.6712111        test: 0.6798494 best: 0.6798991 (1574)  total: 3m 10s   remaining: 26m 31s
1800:   learn: 0.6726767        test: 0.6797910 best: 0.6800546 (1611)  total: 3m 32s   remaining: 25m 56s
2000:   learn: 0.6738319        test: 0.6790730 best: 0.6800546 (1611)  total: 3m 54s   remaining: 25m 24s
bestTest = 0.6800546396
bestIteration = 1611
Shrink model to first 1612 iterations.
[0.6804033402207136, 0.6791561440738999, 0.6802987702706541, 0.6827683102831964, 0.6800546396251115]
0.6805362408947151
[0.7103858706085622, 0.7097606177606178, 0.7087576064522517, 0.7099581770428166, 0.7081182902736726]
0.7093961124275843
PS D:\Study\STUDY>  d:; cd 'd:\Study\STUDY'; & 'C:\ProgramData\Anaconda3\python.exe' 'c:\Users\비트캠프\.vscode\extensions\ms-python.python-2021.12.1559732655\pythonFiles\lib\python\debugpy\launcher' '50834' '--' 'd:\Study\STUDY\keras_Dacon\jobcare\Dacon_jobcare_catboost3_test copy 14.py'
- os: Windows-10-10.0.19042-SP0
- python: 3.9.7 (default, Sep 16 2021, 16:59:28) [MSC v.1916 64 bit (AMD64)]
- pandas: 1.3.4
- numpy: 1.19.5
- sklearn: 0.24.2
==================================================
Learning rate set to 0.013941
0:      learn: 0.6094207        test: 0.6131333 best: 0.6131333 (0)     total: 153ms    remaining: 38m 21s
200:    learn: 0.6409880        test: 0.6521999 best: 0.6521999 (200)   total: 26.4s    remaining: 32m 21s
400:    learn: 0.6521279        test: 0.6686936 best: 0.6687914 (398)   total: 51s      remaining: 30m 55s
600:    learn: 0.6577744        test: 0.6725498 best: 0.6725783 (586)   total: 1m 14s   remaining: 29m 51s
800:    learn: 0.6615414        test: 0.6753472 best: 0.6753716 (799)   total: 1m 37s   remaining: 28m 54s
1000:   learn: 0.6646578        test: 0.6783368 best: 0.6785968 (977)   total: 2m 1s    remaining: 28m 15s
1200:   learn: 0.6669238        test: 0.6796377 best: 0.6796753 (1199)  total: 2m 24s   remaining: 27m 41s
1400:   learn: 0.6688686        test: 0.6794879 best: 0.6800272 (1330)  total: 2m 47s   remaining: 27m 3s
1600:   learn: 0.6705959        test: 0.6795718 best: 0.6800272 (1330)  total: 3m 9s    remaining: 26m 25s
1800:   learn: 0.6718774        test: 0.6796919 best: 0.6802459 (1778)  total: 3m 31s   remaining: 25m 53s
2000:   learn: 0.6730464        test: 0.6799679 best: 0.6802459 (1778)  total: 3m 54s   remaining: 25m 21s
2200:   learn: 0.6739630        test: 0.6792366 best: 0.6802459 (1778)  total: 4m 16s   remaining: 24m 51s
2400:   learn: 0.6748243        test: 0.6794704 best: 0.6802459 (1778)  total: 4m 38s   remaining: 24m 22s
2600:   learn: 0.6756568        test: 0.6795032 best: 0.6802459 (1778)  total: 5m 1s    remaining: 23m 55s
bestTest = 0.6802459134
bestIteration = 1778
Shrink model to first 1779 iterations.
==================================================
Learning rate set to 0.013941
0:      learn: 0.6355245        test: 0.6312463 best: 0.6312463 (0)     total: 150ms    remaining: 37m 27s
200:    learn: 0.6412801        test: 0.6485143 best: 0.6485143 (200)   total: 26.8s    remaining: 32m 56s
400:    learn: 0.6519832        test: 0.6648176 best: 0.6649041 (398)   total: 51.7s    remaining: 31m 23s
600:    learn: 0.6570655        test: 0.6704354 best: 0.6704477 (599)   total: 1m 16s   remaining: 30m 20s
800:    learn: 0.6610953        test: 0.6737213 best: 0.6737817 (794)   total: 1m 39s   remaining: 29m 18s
1000:   learn: 0.6643847        test: 0.6754659 best: 0.6755042 (999)   total: 2m 2s    remaining: 28m 36s
1200:   learn: 0.6668298        test: 0.6765828 best: 0.6766381 (1166)  total: 2m 25s   remaining: 27m 51s
1400:   learn: 0.6690469        test: 0.6779276 best: 0.6779338 (1399)  total: 2m 48s   remaining: 27m 12s
1600:   learn: 0.6709502        test: 0.6783576 best: 0.6783576 (1600)  total: 3m 10s   remaining: 26m 35s
1800:   learn: 0.6723239        test: 0.6787117 best: 0.6789297 (1723)  total: 3m 33s   remaining: 26m 1s
2000:   learn: 0.6736520        test: 0.6786739 best: 0.6789297 (1723)  total: 3m 55s   remaining: 25m 30s
2200:   learn: 0.6748203        test: 0.6782280 best: 0.6789297 (1723)  total: 4m 17s   remaining: 24m 59s
2400:   learn: 0.6756649        test: 0.6779848 best: 0.6789297 (1723)  total: 4m 39s   remaining: 24m 28s
2600:   learn: 0.6766214        test: 0.6772240 best: 0.6789297 (1723)  total: 5m 1s    remaining: 23m 58s
bestTest = 0.6789297017
bestIteration = 1723
Shrink model to first 1724 iterations.
==================================================
Learning rate set to 0.013941
0:      learn: 0.6193509        test: 0.6224305 best: 0.6224305 (0)     total: 153ms    remaining: 38m 9s
200:    learn: 0.6401091        test: 0.6519125 best: 0.6520074 (199)   total: 26.1s    remaining: 32m 3s
400:    learn: 0.6505877        test: 0.6659984 best: 0.6660542 (393)   total: 50.7s    remaining: 30m 45s
600:    learn: 0.6565229        test: 0.6717123 best: 0.6717656 (567)   total: 1m 14s   remaining: 29m 39s
800:    learn: 0.6611631        test: 0.6751187 best: 0.6753960 (799)   total: 1m 37s   remaining: 28m 45s
1000:   learn: 0.6646246        test: 0.6781602 best: 0.6781722 (998)   total: 2m       remaining: 27m 59s
1200:   learn: 0.6669027        test: 0.6792069 best: 0.6792911 (1196)  total: 2m 22s   remaining: 27m 22s
1400:   learn: 0.6689756        test: 0.6795061 best: 0.6796184 (1249)  total: 2m 45s   remaining: 26m 42s
1600:   learn: 0.6705656        test: 0.6796997 best: 0.6799088 (1585)  total: 3m 7s    remaining: 26m 8s
1800:   learn: 0.6720713        test: 0.6800798 best: 0.6801778 (1793)  total: 3m 29s   remaining: 25m 36s
2000:   learn: 0.6731372        test: 0.6800624 best: 0.6803302 (1913)  total: 3m 51s   remaining: 25m 5s
2200:   learn: 0.6741751        test: 0.6800257 best: 0.6803302 (1913)  total: 4m 13s   remaining: 24m 36s
2400:   learn: 0.6751456        test: 0.6795868 best: 0.6803302 (1913)  total: 4m 35s   remaining: 24m 6s
2600:   learn: 0.6759062        test: 0.6797602 best: 0.6803302 (1913)  total: 4m 57s   remaining: 23m 38s
2800:   learn: 0.6767497        test: 0.6800736 best: 0.6803302 (1913)  total: 5m 19s   remaining: 23m 12s
bestTest = 0.6803302184
bestIteration = 1913
Shrink model to first 1914 iterations.
==================================================
Learning rate set to 0.013941
0:      learn: 0.6091699        test: 0.6117842 best: 0.6117842 (0)     total: 149ms    remaining: 37m 19s
200:    learn: 0.6416151        test: 0.6527523 best: 0.6527681 (199)   total: 25.7s    remaining: 31m 35s
400:    learn: 0.6518610        test: 0.6669430 best: 0.6669859 (399)   total: 50.1s    remaining: 30m 23s
600:    learn: 0.6581510        test: 0.6738927 best: 0.6739361 (596)   total: 1m 13s   remaining: 29m 16s
800:    learn: 0.6618129        test: 0.6778215 best: 0.6779038 (798)   total: 1m 36s   remaining: 28m 29s
1000:   learn: 0.6650640        test: 0.6803054 best: 0.6803489 (999)   total: 1m 59s   remaining: 27m 45s
1200:   learn: 0.6675854        test: 0.6812598 best: 0.6814435 (1189)  total: 2m 21s   remaining: 27m 6s
1400:   learn: 0.6695292        test: 0.6820525 best: 0.6823035 (1389)  total: 2m 44s   remaining: 26m 35s
1600:   learn: 0.6711603        test: 0.6829853 best: 0.6830434 (1597)  total: 3m 6s    remaining: 26m 1s
1800:   learn: 0.6721153        test: 0.6828303 best: 0.6832195 (1623)  total: 3m 28s   remaining: 25m 26s
2000:   learn: 0.6731509        test: 0.6825575 best: 0.6832195 (1623)  total: 3m 50s   remaining: 24m 55s
2200:   learn: 0.6741661        test: 0.6825427 best: 0.6832195 (1623)  total: 4m 12s   remaining: 24m 27s
2400:   learn: 0.6750359        test: 0.6824650 best: 0.6832195 (1623)  total: 4m 34s   remaining: 23m 59s
2600:   learn: 0.6757549        test: 0.6824708 best: 0.6832195 (1623)  total: 4m 56s   remaining: 23m 32s
bestTest = 0.6832195206
bestIteration = 1623
Shrink model to first 1624 iterations.
==================================================
Learning rate set to 0.013941
0:      learn: 0.6243277        test: 0.6210133 best: 0.6210133 (0)     total: 153ms    remaining: 38m 21s
200:    learn: 0.6412986        test: 0.6508233 best: 0.6508233 (200)   total: 26.1s    remaining: 32m 5s
400:    learn: 0.6514146        test: 0.6667217 best: 0.6667461 (398)   total: 50.4s    remaining: 30m 33s
600:    learn: 0.6573692        test: 0.6731169 best: 0.6732531 (595)   total: 1m 14s   remaining: 29m 36s
800:    learn: 0.6615658        test: 0.6765628 best: 0.6766486 (799)   total: 1m 37s   remaining: 28m 44s
1000:   learn: 0.6648190        test: 0.6783087 best: 0.6783692 (999)   total: 2m       remaining: 28m 6s
1200:   learn: 0.6675045        test: 0.6786685 best: 0.6789326 (1193)  total: 2m 23s   remaining: 27m 27s
1400:   learn: 0.6695119        test: 0.6795092 best: 0.6795831 (1399)  total: 2m 46s   remaining: 26m 57s
1600:   learn: 0.6712431        test: 0.6802104 best: 0.6802713 (1598)  total: 3m 8s    remaining: 26m 19s
1800:   learn: 0.6726369        test: 0.6801400 best: 0.6802713 (1598)  total: 3m 30s   remaining: 25m 44s
2000:   learn: 0.6739808        test: 0.6793728 best: 0.6802713 (1598)  total: 3m 52s   remaining: 25m 11s
2200:   learn: 0.6750551        test: 0.6792203 best: 0.6802713 (1598)  total: 4m 14s   remaining: 24m 39s
2400:   learn: 0.6758940        test: 0.6793716 best: 0.6802713 (1598)  total: 4m 36s   remaining: 24m 9s
bestTest = 0.6802713355
bestIteration = 1598
Shrink model to first 1599 iterations.
[0.6802459133584331, 0.678929701735878, 0.6803302184257295, 0.6832195205577302, 0.6802713354555101]
0.6805993379066562
[0.7105382423204257, 0.7093458663995684, 0.7114488051845974, 0.7090995609860927, 0.7079950724216293]
0.7096855094624628
'''





