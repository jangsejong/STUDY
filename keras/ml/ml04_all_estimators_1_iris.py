from sklearn.utils import all_estimators #회기 R2 분류 acc
from sklearn.metrics import accuracy_score
from sklearn import datasets
import warnings
warnings.filterwarnings('ignore')


import numpy as np
from sklearn.datasets import load_iris
from tensorflow.python.keras.metrics import accuracy


#1. 데이터
datasets= load_iris()
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
AdaBoostClassifier 의 정답율 : 0.6333333333333333
BaggingClassifier 의 정답율 : 0.9333333333333333
BernoulliNB 의 정답율 : 0.4
CalibratedClassifierCV 의 정답율 : 0.9666666666666667
CategoricalNB 의 정답율 : 0.3333333333333333
ComplementNB 의 정답율 : 0.6666666666666666
DecisionTreeClassifier 의 정답율 : 0.9333333333333333
DummyClassifier 의 정답율 : 0.3
ExtraTreeClassifier 의 정답율 : 0.7
ExtraTreesClassifier 의 정답율 : 0.9333333333333333
GaussianNB 의 정답율 : 0.9666666666666667
GaussianProcessClassifier 의 정답율 : 0.9666666666666667
GradientBoostingClassifier 의 정답율 : 0.9333333333333333
HistGradientBoostingClassifier 의 정답율 : 0.8666666666666667
KNeighborsClassifier 의 정답율 : 1.0
LabelPropagation 의 정답율 : 0.9666666666666667
LabelSpreading 의 정답율 : 0.9666666666666667
LinearDiscriminantAnalysis 의 정답율 : 1.0
LinearSVC 의 정답율 : 0.9666666666666667
LogisticRegression 의 정답율 : 0.9666666666666667
LogisticRegressionCV 의 정답율 : 1.0
MLPClassifier 의 정답율 : 0.9
MultinomialNB 의 정답율 : 0.6333333333333333
NearestCentroid 의 정답율 : 0.9666666666666667
NuSVC 의 정답율 : 0.9666666666666667
PassiveAggressiveClassifier 의 정답율 : 0.7666666666666667
Perceptron 의 정답율 : 0.9333333333333333
QuadraticDiscriminantAnalysis 의 정답율 : 1.0
RadiusNeighborsClassifier 의 정답율 : 0.4666666666666667
RandomForestClassifier 의 정답율 : 0.9666666666666667
RidgeClassifier 의 정답율 : 0.9333333333333333
RidgeClassifierCV 의 정답율 : 0.8333333333333334
SGDClassifier 의 정답율 : 0.9
SVC 의 정답율 : 1.0
'''