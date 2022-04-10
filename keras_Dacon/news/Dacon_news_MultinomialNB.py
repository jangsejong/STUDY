import pandas as pd
import numpy as np
import os
from glob import glob

path = "D:\\Study\\_data\\dacon\\news\\"
train = pd.read_csv(path+ 'train.csv')
test = pd.read_csv(path +"test.csv") #파일 읽기

print(train.keys())
print(train.isnull().values.any())

train_email = train['text'].values
train_label = train['target'].values
test_email = test['text'].values
test_label = test['target'].values



