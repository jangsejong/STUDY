from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

#1. 데이터
docs = ['너무 재밌어요', ' 참 최고에요', ' 참 잘 만든 영화에요', '추천하고 싶은 영화입니다', '한 번 더 보고 싶네요', ' 글쎄요', ' 별로에요', ' 생각보다 지루해요', ' 연기가 어색해요'
        , ' 재미없어요', ' 너무 재미없어요', ' 참 재밌네요', ' 예람이가 잘 생기긴 했어요']



labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1])

token = Tokenizer()
token.fit_on_texts(docs)
# print(token.word_index)

# {'참': 1, '너무': 2, '잘': 3, '재미없어요': 4, '재밌어요': 5, '최고에요': 6, '만든': 7, '영화에요': 8, '추천하고': 9, 
#  '싶은': 10, '영화입니다': 11, '한': 12, '번': 13, '더': 14, '보고': 15, '싶네요': 16, '글쎄요': 17, '별로에요': 18, 
#  '생각보다': 19, '지루해요': 20, '연기가': 21, '어색해요': 22, '재밌네요': 23, '예람이가': 24, '생기긴': 25, '했어요': 26}

x = token.texts_to_sequences(docs)
# print(x)
# [[2, 5], [1, 6], [1, 3, 7, 8], [9, 10, 11], [12, 13, 14, 15, 16], [17], 
#  [18], [19, 20], [21, 22], [4], [2, 4], [1, 23], [24, 3, 25, 26]]

from tensorflow.keras.preprocessing.sequence import *
pad_x = pad_sequences(x, padding='pre', maxlen=5)
# print(pad_x)
# print(pad_x.shape)  #(13, 5)
world_size = len(token.word_index)
print("world_size :", world_size)   #26

print(np.unique(pad_x))
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26]
#이 리스트를 numpy.unique() 함수에 넣어주면 다음과 같이 유니크한 값들만 오름차순으로 나열된 넘파이 배열(numpy.ndarray)이 반환됩니다.

