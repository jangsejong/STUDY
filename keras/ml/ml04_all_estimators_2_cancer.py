from sklearn.utils import all_estimators #회기 R2 분류 acc
from sklearn.metrics import accuracy_score
from sklearn import datasets
import warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import load_iris
from tensorflow.python.keras.metrics import accuracy
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler


#1. 데이타 
datasets = load_breast_cancer()
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
AdaBoostClassifier 의 정답율 : 0.9473684210526315
BaggingClassifier 의 정답율 : 0.956140350877193
BernoulliNB 의 정답율 : 0.6403508771929824
CalibratedClassifierCV 의 정답율 : 0.9649122807017544
ComplementNB 의 정답율 : 0.7807017543859649
DecisionTreeClassifier 의 정답율 : 0.9298245614035088
DummyClassifier 의 정답율 : 0.6403508771929824
ExtraTreeClassifier 의 정답율 : 0.9122807017543859
ExtraTreesClassifier 의 정답율 : 0.956140350877193
GaussianNB 의 정답율 : 0.9210526315789473
GaussianProcessClassifier 의 정답율 : 0.9649122807017544
GradientBoostingClassifier 의 정답율 : 0.956140350877193
HistGradientBoostingClassifier 의 정답율 : 0.9736842105263158
KNeighborsClassifier 의 정답율 : 0.956140350877193
LabelPropagation 의 정답율 : 0.9473684210526315
LabelSpreading 의 정답율 : 0.9473684210526315
LinearDiscriminantAnalysis 의 정답율 : 0.9473684210526315
LinearSVC 의 정답율 : 0.9736842105263158
LogisticRegression 의 정답율 : 0.9649122807017544
LogisticRegressionCV 의 정답율 : 0.9736842105263158
MLPClassifier 의 정답율 : 0.9649122807017544
MultinomialNB 의 정답율 : 0.8508771929824561
NearestCentroid 의 정답율 : 0.9298245614035088
NuSVC 의 정답율 : 0.9473684210526315
PassiveAggressiveClassifier 의 정답율 : 0.9736842105263158
Perceptron 의 정답율 : 0.9736842105263158
QuadraticDiscriminantAnalysis 의 정답율 : 0.9385964912280702
RandomForestClassifier 의 정답율 : 0.9824561403508771
RidgeClassifier 의 정답율 : 0.9473684210526315
RidgeClassifierCV 의 정답율 : 0.9473684210526315
SGDClassifier 의 정답율 : 0.9824561403508771
SVC 의 정답율 : 0.9736842105263158
'''