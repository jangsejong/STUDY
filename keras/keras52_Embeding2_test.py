import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request
# from konlpy.tag import Okt
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


#1) 데이터 로드하기

urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")

train_data = pd.read_table('ratings_train.txt')
test_data = pd.read_table('ratings_test.txt')

print('훈련용 리뷰 개수 :',len(train_data)) # 훈련용 리뷰 개수 출력

# 훈련용 리뷰 개수 : 150000

train_data[:5] # 상위 5개 출력

print('테스트용 리뷰 개수 :',len(test_data)) # 테스트용 리뷰 개수 출력

# 테스트용 리뷰 개수 : 50000

test_data[:5]

#2) 데이터 정제하기

# document 열과 label 열의 중복을 제외한 값의 개수
train_data['document'].nunique(), train_data['label'].nunique()    #(146182, 2)

# document 열의 중복 제거
train_data.drop_duplicates(subset=['document'], inplace=True)

print('총 샘플의 수 :',len(train_data))

train_data['label'].value_counts().plot(kind = 'bar')

print(train_data.groupby('label').size().reset_index(name = 'count'))

print(train_data.isnull().values.any())

True

print(train_data.isnull().sum())
train_data.loc[train_data.document.isnull()]

train_data = train_data.dropna(how = 'any') # Null 값이 존재하는 행 제거
print(train_data.isnull().values.any()) # Null 값이 존재하는지 확인



