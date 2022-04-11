import pandas as pd
import numpy as np
import os
from glob import glob

path = "D:\\Study\\_data\\dacon\\news\\"
train = pd.read_csv(path+ 'train.csv')
test = pd.read_csv(path +"test.csv") #파일 읽기
submission = pd.read_csv(path +"sample_submission.csv")

# train.head()
# test.head()
# submission.head()

import re 
# 숫자와 영문 빼고 모두 제거
def clean_text(texts): 
  corpus = [] 
  for i in range(0, len(texts)): 

    review = re.sub(r'[@%\\*=()/~#&\+á?\xc3\xa1\-\|\.\:\;\!\-\,\_\~\$\'\"\n\]\[\>]', '',texts[i]) #@%*=()/+ 와 같은 문장부호 제거
    review = re.sub(r'\s+', ' ', review) #extra space 제거
    review = re.sub(r'<[^>]+>','',review) #Html tags 제거
    review = re.sub(r'\s+', ' ', review) #spaces 제거
    review = re.sub(r"^\s+", '', review) #space from start 제거
    review = re.sub(r'\s+$', '', review) #space from the end 제거
    review = re.sub(r'_', ' ', review) #space from the end 제거
    corpus.append(review) 
  
  return corpus

temp1 = clean_text(train['text']) #메소드 적용
train['text'] = temp1
train.head()

temp2 = clean_text(test['text']) #메소드 적용
test['text'] = temp2
test.head()

# from sklearn.model_selection import train_test_split
# train_dataset, val_dataset = train_test_split(train, test_size = 0.1)
# print(len(train_dataset))
# print(len(val_dataset))

# # dataloader에서 오류가 나서 인덱스 재설정
# train_dataset = train_dataset.reset_index(drop=True)
# val_dataset = val_dataset.reset_index(drop=True)



from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

vectorizer = TfidfVectorizer(ngram_range=(1, 2))

vectorizer.fit(np.array(train["text"]))

train_vec = vectorizer.transform(train["text"])
train_y = train["target"]

test_vec = vectorizer.transform(test["text"])

from sklearn.neural_network import MLPClassifier

# model = MLPClassifier()
# model.fit(train_vec, train_y)

model = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(100,), random_state=66)
model.fit(train_vec, train_y)

pred = model.predict(test_vec)


submission["target"] = pred

submission.to_csv(path+ "0411_01.csv", index = False)


