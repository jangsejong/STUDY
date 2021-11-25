from sklearn.datasets import load_boston
datasets = load_boston()



#1. 데이터
x = datasets.data
y = datasets.target
print(x)
print(y)
print(x.shape)
print(y.shape)

print(datasets.feature_names)
print(datasets.DESCR)