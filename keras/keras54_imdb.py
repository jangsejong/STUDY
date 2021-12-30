import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request
# from konlpy.tag import Okt
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

(x_train, y_train), (x_test, y_test) = imdb.load_data(
        num_words=10000
)

print(x_train, len(x_train),len(x_test))    #8982,2246
print(y_train[0])    #3
print(np.unique(y_train))

# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
#  24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]


print(type(x_train), type(y_train))   
#<class 'numpy.ndarray'> <class 'numpy.ndarray'>

print(len(x_train[0]), len(x_train[1]))   #87 56


print(type(x_train[0]), type(x_train[1]))   #<class 'list'> <class 'list'>.


print("뉴스기사의 최대길이 :", max(len(i) for i in x_train))
print("뉴스기사의 평균길이 :", sum(map(len, x_train))/len(x_train))

# 영화리뷰의 최대길이 : 2494
# 영화리뷰의 평균길이 : 238.71364

# 전처리
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

x_train = pad_sequences(x_train, padding='pre', maxlen=100, truncating='pre')

x_test = pad_sequences(x_test, padding='pre', maxlen=100, truncating='pre')

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)



from tensorflow.keras.models import Sequential, load_model, Model,save_model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout, Embedding, LSTM


word_to_index = imdb.get_word_index()
# print(word_to_index)
# print(sorted(word_to_index.items())) # 키 위주로
import operator
print(sorted(word_to_index.items(), key=operator.itemgetter(1)))

#키벨류 데이터셋 예제 3개

index_to_word = {}
for key, value in word_to_index.items():
    index_to_word[value+3] = key

for index, token in enumerate(("<pad>", "<sos>", "<unk>")):
      index_to_word[index] = token

print(' '.join([index_to_word[index] for index in x_train[0]]))

#Link : https://wikidocs.net/22933







#2. 모델
model = Sequential()











