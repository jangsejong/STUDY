# Importing data - train, test and submission
import pandas as pd
pd.set_option('display.max_columns', None)
train = pd.read_csv('../input/edsa-individual-electricity-shortfall-challenge/df_train.csv')
test = pd.read_csv('../input/edsa-individual-electricity-shortfall-challenge/df_test.csv')
submission = pd.read_csv('../input/edsa-individual-electricity-shortfall-challenge/sample_submission_load_shortfall (1).csv')

# Getting columns
y = train.iloc[:,-1].values
# y
# Getting column for submussion
time = test.iloc[:,1].values
# time

EDA and Data Cleaning 
# Dropping identification columns
train.drop(columns = ['Unnamed: 0', 'time'], inplace = True)
test.drop(columns = ['Unnamed: 0', 'time'], inplace = True)
# Correcting types
train.dtypes
test.dtypes
# Getting null precentages where there is null
train.isnull().sum()[train.isnull().sum() != 0].sort_values(ascending=False) *100 / train.shape[0]
test.isnull().sum()[test.isnull().sum() != 0].sort_values(ascending=False) *100 / test.shape[0]
# Shape of data
counts = train.iloc[:,:-1].nunique()

# counts
# Shape of data
counts = test.iloc[:,:-1].nunique()

# counts

train.duplicated().sum() # duplicate rows
test.duplicated().sum() # duplicate rows
# Getting mean and standrad deviation
train.describe()
# Getting mean and standrad deviation
test.describe()
# Frequency distribution of all columns - visual check if any value is more than 85%
import matplotlib.pyplot as plt
count = 0
fig, ax = plt.subplots(10, 5, figsize = (18, 20))
for i in range(train.shape[1]):
    plt.subplot(10, 5, count + 1)
    plt.hist(train.iloc[:, i].dropna(axis = 0), rwidth = 0.9, color = 'green')
    plt.xlabel(train.columns[i], fontsize = 15)
    count += 1
plt.tight_layout()
plt.show()

# Encoding 

# Label Encoding
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
train['Valencia_wind_deg'] = labelencoder.fit_transform(train['Valencia_wind_deg'])
train['Seville_pressure'] = labelencoder.fit_transform(train['Seville_pressure'])
test['Valencia_wind_deg'] = labelencoder.fit_transform(test['Valencia_wind_deg'])
test['Seville_pressure'] = labelencoder.fit_transform(test['Seville_pressure'])
train.drop(columns = ['load_shortfall_3h'], inplace = True)

# Testing Multicollinearity 

# VIF
import numpy as np
inv_corr_matrix = np.linalg.inv(train.corr())
inv_corr_matrix = pd.DataFrame(data = inv_corr_matrix, index = train.columns, columns = train.columns)
# corr = X_df.astype(float).corr()
vif_coefficients =  np.diag(np.array(inv_corr_matrix))
mutlicollinear_column_indices = [i for i in range(len(vif_coefficients)) if vif_coefficients[i] > 10]
# Drop data with non collinear variables
train.drop(train.columns[mutlicollinear_column_indices], axis = 1, inplace = True)
test.drop(test.columns[mutlicollinear_column_indices], axis = 1, inplace = True)
# Filling series with mean or mode
train['Valencia_pressure'].fillna(train['Valencia_pressure'].mean(), inplace = True)
test['Valencia_pressure'].fillna(test['Valencia_pressure'].mean(), inplace = True)

# Model Fitting and Parameter Tuning 

# XGB Regressor
from xgboost import XGBRegressor
regressor = XGBRegressor(learning_rate= 0.05, max_depth= 4) #paramters added after tuning in next step
regressor.fit(train, y)
# Parameter Tuning
from sklearn.model_selection import GridSearchCV
parameters = [{'max_depth': [4, 5, 6, 7, 8], 'learning_rate': [0.01, 0.05, 0.1, 0.15]}]
grid_search = GridSearchCV(estimator = regressor, param_grid = parameters, scoring = 'neg_mean_squared_error', cv = 10, n_jobs= -1)
grid_search = grid_search.fit(X_opt, y)
best_accuracy =  grid_search.best_score_
best_parameters = grid_search.best_params_
print('best_accuracy: ', best_accuracy, 'best_parameters: ', best_parameters)

# Prediction and Submission 

# Prediction
y_pred = regressor.predict(test)
y_pred
# Output / Submitting / Submission
output = pd.DataFrame({'time': time, 'load_shortfall_3h': y_pred})
output.to_csv('submission.csv', index=False)
print("Submitted successfully!")








