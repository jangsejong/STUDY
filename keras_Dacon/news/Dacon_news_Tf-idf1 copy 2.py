import pandas as pd
import numpy as np
import os
from glob import glob

path = "D:\\Study\\_data\\dacon\\news\\"
train = pd.read_csv(path+ 'train.csv')
test = pd.read_csv(path +"test.csv") #파일 읽기
submission = pd.read_csv(path +"sample_submission.csv")

train.head()
test.head()
submission.head()

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

vectorizer = TfidfVectorizer(ngram_range=(1, 2))

vectorizer.fit(np.array(train["text"]))

train_vec = vectorizer.transform(train["text"])
train_y = train["target"]

test_vec = vectorizer.transform(test["text"])

from sklearn.neural_network import MLPClassifier

# model = MLPClassifier()
# model.fit(train_vec, train_y)

model = MLPClassifier()
model.fit(train_vec, train_y)

pred = model.predict(test_vec)


submission["target"] = pred

submission.to_csv(path+ "0408_03.csv", index = False)


