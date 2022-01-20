import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor

import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_wine, load_diabetes, load_boston, load_breast_cancer, fetch_covtype

datasets = {'Iris':load_iris(),
            'Wine':load_wine(),
            'Diabets':load_diabetes(),
            'Cancer':load_breast_cancer(),
            'Boston':load_boston(),
            'FetchCov':fetch_covtype(),
            'Kaggle_Bike':'Kaggle_Bike'
            }

model_1 = DecisionTreeClassifier(random_state=66, max_depth=5)
model_1r = DecisionTreeRegressor(random_state=66, max_depth=5)

model_2 = RandomForestClassifier(random_state=66, max_depth=5)
model_2r = RandomForestRegressor(random_state=66, max_depth=5)

model_3 = XGBClassifier(random_state=66)
model_3r = XGBRegressor(random_state=66)

model_4 = GradientBoostingClassifier(random_state=66)
model_4r = GradientBoostingRegressor(random_state=66)

def plot_feature_importances_dataset(model):
    n_features = dataset.data.shape[1]
    plt.barh(np.arange(n_features),model.feature_importances_,align='center')
    plt.yticks(np.arange(n_features), dataset.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1,n_features)

path = "D:\\Study\\_data\\kaggle\\bike\\"
train = pd.read_csv(path + "train.csv")

model_list = [model_1,model_2,model_3,model_4]
model_list_r = [model_1r,model_2r,model_3r,model_4r]

model_name = ['DecisionTree','RandomForest','XGB','GradientBoosting']

for (dataset_name, dataset) in datasets.items():
    print(f'------------{dataset_name}-----------')
    print('====================================')    
    
    if dataset_name == 'Kaggle_Bike':
        y = train['count']
        x = train.drop(['casual', 'registered', 'count'], axis=1)        
        x['datetime'] = pd.to_datetime(x['datetime'])
        x['year'] = x['datetime'].dt.year
        x['month'] = x['datetime'].dt.month
        x['day'] = x['datetime'].dt.day
        x['hour'] = x['datetime'].dt.hour
        x = x.drop('datetime', axis=1)
        y = np.log1p(y)        
    else:
        x = dataset.data
        y = dataset.target    

    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        train_size=0.8, shuffle=True, random_state=66)
    
    plt.figure(figsize=(15,10))
    for i in range(4):        
        plt.subplot(2, 2, i+1)               # nrows=2, ncols=1, index=1
        if dataset_name == 'Cancer':
            model_list_r[i].fit(x_train, y_train)
            score = model_list_r[i].score(x_test, y_test)
            feature_importances_ = model_list_r[i].feature_importances_

            # y_predict = model_list[i].predict(x_test)
            # acc = accuracy_score(y_test, y_predict)
            print("score", score)
            # print("accuracy_score", acc)
            print("feature_importances_", feature_importances_)
            plot_feature_importances_dataset(model_list_r[i])    
            
        else: 
            model_list[i].fit(x_train, y_train)
            score = model_list[i].score(x_test, y_test)
            feature_importances_ = model_list[i].feature_importances_

            # y_predict = model_list[i].predict(x_test)
            # acc = accuracy_score(y_test, y_predict)
            print("score", score)
            # print("accuracy_score", acc)
            print("feature_importances_", feature_importances_)
            plot_feature_importances_dataset(model_list[i])    
            plt.ylabel(model_name[i])
            plt.title(dataset_name)

    plt.tight_layout()
    plt.show()
    
    
    '''
------------Iris-----------
====================================
score 0.9666666666666667
feature_importances_ [0.         0.0125026  0.53835801 0.44913938]
score 0.9666666666666667
feature_importances_ [0.08150824 0.02190985 0.46987909 0.42670282]
C:\ProgramData\Anaconda3\lib\site-packages\xgboost\sklearn.py:1224: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].     
  warnings.warn(label_encoder_deprecation_msg, UserWarning)
[11:57:41] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.5.1/src/learner.cc:1115: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'multi:softprob' was changed from 'merror' to 'mlogloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
score 0.9
feature_importances_ [0.01835513 0.0256969  0.6204526  0.33549538]
score 0.9666666666666667
feature_importances_ [0.00226872 0.01356986 0.38159741 0.60256401]
------------Wine-----------
====================================
score 0.9444444444444444
feature_importances_ [0.01598859 0.00489447 0.         0.         0.         0.
 0.1569445  0.         0.         0.04078249 0.08604186 0.33215293
 0.36319516]
score 1.0
feature_importances_ [0.14074602 0.02731994 0.01584483 0.04709609 0.02201035 0.06085119
 0.1769038  0.01467615 0.02739868 0.12602185 0.07531074 0.1187375
 0.14708285]
C:\ProgramData\Anaconda3\lib\site-packages\xgboost\sklearn.py:1224: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].     
  warnings.warn(label_encoder_deprecation_msg, UserWarning)
[11:57:45] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.5.1/src/learner.cc:1115: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'multi:softprob' was changed from 'merror' to 'mlogloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
score 1.0
feature_importances_ [0.01854127 0.04139537 0.01352911 0.01686821 0.02422602 0.00758254
 0.10707159 0.01631111 0.00051476 0.12775213 0.01918284 0.50344414
 0.10358089]
score 0.9722222222222222
feature_importances_ [1.49470796e-02 4.16158823e-02 2.66374739e-02 3.35194115e-03
 2.50709383e-03 3.48521584e-05 1.06037687e-01 1.26209560e-04
 1.66667944e-04 2.50956056e-01 2.98140736e-02 2.48782846e-01
 2.75022137e-01]
------------Diabets-----------
====================================
score 0.0
feature_importances_ [0.13704006 0.         0.06882606 0.27482734 0.06316102 0.
 0.21526266 0.03437816 0.03867543 0.16782927]
score 0.0
feature_importances_ [0.11426269 0.0151792  0.13541595 0.11582904 0.09538985 0.08814125
 0.13857309 0.06612734 0.12962632 0.10145528]
C:\ProgramData\Anaconda3\lib\site-packages\xgboost\sklearn.py:1224: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].     
  warnings.warn(label_encoder_deprecation_msg, UserWarning)
[11:57:48] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.5.1/src/learner.cc:1115: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'multi:softprob' was changed from 'merror' to 'mlogloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
score 0.011235955056179775
feature_importances_ [0.11070196 0.09295782 0.10416857 0.10149716 0.08342394 0.08078491
 0.12378592 0.09546181 0.1062927  0.10092521]
score 0.0
feature_importances_ [0.10276755 0.02109063 0.12274576 0.10423223 0.11832427 0.14053768
 0.10299438 0.03837181 0.13239798 0.11653772]
------------Cancer-----------
====================================
score 0.6095890410958902
feature_importances_ [0.         0.06054151 0.         0.         0.         0.
 0.         0.02005078 0.         0.02291518 0.         0.
 0.         0.01973513 0.         0.         0.00636533 0.00442037
 0.         0.004774   0.         0.01642816 0.         0.72839202
 0.         0.         0.00470676 0.11167078 0.         0.        ]
score 0.8430773351776508
feature_importances_ [0.00164779 0.0257168  0.00218836 0.00557117 0.00245026 0.00183587
 0.00175095 0.08303949 0.0033565  0.00199656 0.00556397 0.00233348
 0.0031882  0.01329712 0.00397206 0.00217954 0.0025278  0.00106078
 0.00137761 0.00211498 0.11941773 0.02072063 0.28104511 0.18753447
 0.00670488 0.00340266 0.01265381 0.19731097 0.0018223  0.00221814]
score 0.8186041948790475
feature_importances_ [3.3426348e-03 1.5599440e-02 5.5710827e-03 3.8845095e-04 7.7899033e-03
 1.2560452e-02 4.5806501e-04 2.4379442e-02 9.8811928e-04 4.4432459e-06
 7.2003021e-03 7.9250028e-03 6.7853391e-05 7.0645725e-03 5.3733634e-04
 9.1388347e-03 2.1158117e-04 4.4393907e-03 2.1952372e-03 5.9370400e-04
 4.0454051e-01 3.9285924e-02 3.3381354e-04 3.5936674e-01 2.2880486e-05
 6.7483838e-06 1.4275628e-02 7.1571201e-02 1.3233325e-04 8.2272045e-06]
score 0.8370799295445284
feature_importances_ [2.24990072e-04 3.66567877e-02 1.37776651e-03 3.12169237e-03
 1.72102934e-04 4.43302759e-03 8.23225138e-04 1.62740676e-01
 1.40744105e-03 4.39347661e-04 3.90223901e-03 3.21431418e-04
 1.51138536e-03 1.77029870e-02 1.03112690e-03 4.24392360e-03
 7.20455539e-03 5.15831697e-04 9.32780547e-05 3.81976890e-03
 3.02510477e-01 4.15241559e-02 5.31834726e-03 2.86867638e-01
 2.08106382e-03 1.90409565e-04 1.25557069e-02 9.39089784e-02
 2.52493793e-04 3.04714557e-03]
------------Boston-----------
====================================
Traceback (most recent call last):
  File "d:\Study\STUDY\machine_running\ml14_FI_1_iris.py", line 86, in <module>
    model_list[i].fit(x_train, y_train)
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\tree\_classes.py", line 903, in fit
    super().fit(
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\tree\_classes.py", line 191, in fit
    check_classification_targets(y)
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\utils\multiclass.py", line 183, in check_classification_targets
    raise ValueError("Unknown label type: %r" % y_type)
ValueError: Unknown label type: 'continuous'

    '''