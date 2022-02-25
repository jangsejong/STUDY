import datasets
from sklearn.datasets import load_boston#, fatch_california_housing
from sklearn.model_selection import train_test_split

datasets = load_boston()
#datasets = fatch_california_housing()

x = datasets.data
y = datasets.target

print(datasets.feature_names)
print(datasets.DESCR)

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=66, train_size=0.8)

#model = LinearRegression()\
model = make_pipeline(LinearRegression())

model.fit(x_train, y_train)

print(model.score(x_test, y_test)) # 0.8111288663608656 
                                   # 0.8111288663608656
  
from sklearn.model_selection import KFold, cross_val_score
scores = cross_val_score(model, x_train, y_train, cv=5, scoring='neg_mean_squared_error')
print(scores)


from sklearn.preprocessing import PolynomialFeatures

pf = PolynomialFeatures(degree=2)
xp = pf.fit_transform(x_train)
print(xp.shape)

