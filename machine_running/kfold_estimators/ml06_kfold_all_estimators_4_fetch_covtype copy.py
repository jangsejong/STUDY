from sklearn.utils import all_estimators #회기 R2 분류 acc
from sklearn.metrics import accuracy_score
from sklearn import datasets
import warnings
warnings.filterwarnings('ignore')


import numpy as np
from sklearn.datasets import fetch_covtype
from tensorflow.python.keras.metrics import accuracy


#1. 데이터
datasets= fetch_covtype()
#print(datasets.DESCR)
x = datasets.data
y = datasets.target


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, shuffle=True, random_state=66)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#모델 구성

allAlgorithms = all_estimators(type_filter='classifier')
# allAlgorithms = all_estimators(type_filter='regressor')

print("allAlgorithms :", allAlgorithms)
print("모델의 갯수 :", len(allAlgorithms))


for (name, algorithm) in allAlgorithms:
  try: 
    model = algorithm()
    model.fit(x_train, y_train)
        
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(name, '의 정답율 :', acc)
    
  except:
    continue



'''
AdaBoostClassifier 의 정답율 : 0.5028613719095032
BaggingClassifier 의 정답율 : 0.961825426193816
BernoulliNB 의 정답율 : 0.631833945767321
CalibratedClassifierCV 의 정답율 : 0.7122621619063191
CategoricalNB 의 정답율 : 0.6321437484402296
ComplementNB 의 정답율 : 0.6225742880992745
DecisionTreeClassifier 의 정답율 : 0.9394593943357744
DummyClassifier 의 정답율 : 0.48625250638968015
ExtraTreeClassifier 의 정답율 : 0.8726366789153465
ExtraTreesClassifier 의 정답율 : 0.9543299226353881
GaussianNB 의 정답율 : 0.09079800005163378
GradientBoostingClassifier 의 정답율 : 0.773491217954786
HistGradientBoostingClassifier 의 정답율 : 0.781993580200167
KNeighborsClassifier 의 정답율 : 0.9376263951877318
LinearDiscriminantAnalysis 의 정답율 : 0.6797931206595355
LinearSVC 의 정답율 : 0.7124170632427734
LogisticRegression 의 정답율 : 0.7194220459024294
LogisticRegressionCV 의 정답율 : 0.7246542688226638
MLPClassifier 의 정답율 : 0.8304949097699715
MultinomialNB 의 정답율 : 0.6410247583969433
NearestCentroid 의 정답율 : 0.38585062347787924
PassiveAggressiveClassifier 의 정답율 : 0.5505107441288091
Perceptron 의 정답율 : 0.6010257910725196
QuadraticDiscriminantAnalysis 의 정답율 : 0.08440401710799204
'''