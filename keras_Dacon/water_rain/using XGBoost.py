# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

data = pd.read_csv('/kaggle/input/rainfall-data/Rainfall1.csv')
data.head()

data.columns

data[['temp', 'realtive_hum', 'specific_hum', 'wind ', 'rainfall']].plot(figsize=(15,6))

from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_log_error

X = data[['temp', 'realtive_hum', 'specific_hum', 'wind ']]
y = data['rainfall']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 12)

model = xgb.XGBRegressor()
model.fit(X_train,y_train)

pred = model.predict(X_test)
RMSLE = np.sqrt( mean_squared_log_error(y_test, pred) )
print("The score is %.5f" % RMSLE )

import matplotlib.pyplot as plt

f = plt.figure()
f.set_figwidth(20)
f.set_figheight(5)
plt.plot(y_test.tolist(),label = 'True Value')
plt.plot(pred,label = 'Predicted Value')
Mape = np.mean(((y_test.tolist()-pred)/pred)*100)
plt.title("Mean Absolute Percentage Error %f " % Mape)
plt.legend(loc='lower right')
plt.show()
