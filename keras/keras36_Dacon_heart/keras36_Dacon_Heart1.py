import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, OneHotEncoder
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

from sklearn import metrics

def f1_score(answer, submission):
    true = answer
    pred = submission
    score = metrics.f1_score(y_true=true, y_pred=pred)
    return score

#1 데이터
path = "D:\\Study\\_data\\dacon\\heart\\"
train = pd.read_csv(path +"train.csv")
test_file = pd.read_csv(path + "test.csv") 
submission = pd.read_csv(path+"sample_submission.csv")

x = train.drop(['id','target'], axis =1)
y = train['target']

print(train.shape, test_file.shape)           # (151, 15) (152, 14)

# import matplotlib.pyplot as plt
# import seaborn as sns

# plt.figure(figsize=(13,13))
# sns.heatmap(data= x.corr(), square=True, annot=True, cbar=True)
# plt.show()    
    

x = x.drop(['chol','fbs','restecg','trestbps'],axis =1, inplace= True)
test_file =test_file.drop(['chol','id','fbs','restecg','trestbps'],axis =1, inplace= True)
# le = LabelEncoder()
# le.fit(train['sex'])
# x['sex'] = le.transform(train['sex'])

# test2 = (test_file[test_file['ca'] == 4]).drop(['ca'], axis = 1)
# test2id = test2['id']

# le.fit(test_file['sex'])
# test_file['sex'] = le.transform(test_file['sex'])

y = y.to_numpy()
x = x.to_numpy()
test_file = test_file.to_numpy()

print(x.shape, test_file.shape)  



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
         train_size = 0.8, shuffle = True, random_state = 66) #

scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = MaxAbsScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model1 = RandomForestClassifier(oob_score= True, bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators= 15000, n_jobs=None, verbose=0, warm_start=False, random_state=66)

model2 = GradientBoostingClassifier(n_estimators = 15000,random_state=66)

model3 = ExtraTreesClassifier(n_estimators = 15000,random_state =66)

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
model4 = HistGradientBoostingClassifier(random_state =66)


#from lightgbm import LGBMClassifier
#model5 = LGBMClassifier(random_state =66)


voting_model = VotingClassifier(estimators=[ ('RandomForestClassifier', model1), ('GradientBoostingClassifier', model2)
                                            ,('ExtraTreesClassifier', model3),('HistGradientBoostingClassifier', model4)], voting='hard')

classifiers = [model1, model2,model3,model4]

for classifier in classifiers:
    classifier.fit(x_train, y_train)
    pred = classifier.predict(x_test)
    class_name = classifier.__class__.__name__
    print("============== " + class_name + " ==================")
    num = accuracy_score(y_test, pred)
    num2 = f1_score(y_test, pred)
    print('{0} 정확도: {1}'.format(class_name, num))
    print('{0} F1_Score : {1}'.format(class_name, num2))
    y_pred_ = classifier.predict(test_file)
    submission['target'] = y_pred_
    submission.to_csv(path+ str(num) +"_" + class_name + "heart1223_001.csv", index=False)
    
voting_model.fit(x_train, y_train)
pred = voting_model.predict(x_test)

# print('===================== 보팅 분류기 ========================')
# num = str(accuracy_score(y_test, pred))
# print('{0} 정확도: {1}'.format(class_name, num))

# y_pred_ = voting_model.predict(test_file)

# submission['target'] = y_pred_
# submission.to_csv(num + "heart001.csv", index=False)

'''
minmax
============== RandomForestClassifier ==================
RandomForestClassifier 정확도: 0.8064516129032258
RandomForestClassifier F1_Score : 0.85
============== GradientBoostingClassifier ==================
GradientBoostingClassifier 정확도: 0.7741935483870968
GradientBoostingClassifier F1_Score : 0.8108108108108109
============== ExtraTreesClassifier ==================
ExtraTreesClassifier 정확도: 0.8709677419354839
ExtraTreesClassifier F1_Score : 0.9
============== HistGradientBoostingClassifier ==================
HistGradientBoostingClassifier 정확도: 0.8387096774193549
HistGradientBoostingClassifier F1_Score : 0.8717948717948718

standard
============== RandomForestClassifier ==================
RandomForestClassifier 정확도: 0.8064516129032258
RandomForestClassifier F1_Score : 0.85
============== GradientBoostingClassifier ==================
GradientBoostingClassifier 정확도: 0.7741935483870968
GradientBoostingClassifier F1_Score : 0.8108108108108109
============== ExtraTreesClassifier ==================
ExtraTreesClassifier 정확도: 0.8709677419354839
ExtraTreesClassifier F1_Score : 0.9
============== HistGradientBoostingClassifier ==================
HistGradientBoostingClassifier 정확도: 0.8387096774193549
HistGradientBoostingClassifier F1_Score : 0.8717948717948718





'''
