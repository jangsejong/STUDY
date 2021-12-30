# from tensorflow.keras.preprocessing import text
'''
from tensorflow.keras.preprocessing.text import Tokenizer

text1 = '나는 진짜 맛있는 밥을 진짜 너무 너무 맛있게 먹었다.'
text2 = '나는 너무 너무 잘생긴 지구용사 태권브이.'


token = Tokenizer()
token.fit_on_texts([text1, text2])

print(token.word_index)  #{'진짜': 1, '너무': 2, '나는': 3, '맛있는': 4, '밥을': 5, '맛있게': 6, '먹었다': 7}

x = token.texts_to_sequences([text1 + text2])
print(x)   #[[1, 2, 5, 6, 2, 3, 3, 7, 8], [1, 4, 4, 9, 10, 11]]
#print(x.shape) list 는 shape 표현하지 않는다. 
import numpy as np


X_train = np.asarray(x)

from tensorflow.keras.utils import to_categorical
word_size = len(token.word_index)
print(word_size)     #11



x = to_categorical(x)
print(x, x.shape)    (1, 15, 11)
'''

from tensorflow.keras.preprocessing.text import Tokenizer

text1 = '나는 진짜 맛있는 밥을 진짜 너무 너무 맛있게 먹었다.'
text2 = '나는 너무 너무 잘생긴 지구용사 태권브이.'


token = Tokenizer()
token.fit_on_texts([text1, text2])

print(token.word_index)  #{'진짜': 1, '너무': 2, '나는': 3, '맛있는': 4, '밥을': 5, '맛있게': 6, '먹었다': 7}

x = token.texts_to_sequences([text1, text2])
print(x)   #[[1, 2, 5, 6, 2, 3, 3, 7, 8], [1, 4, 4, 9, 10, 11]]
#print(x.shape) list 는 shape 표현하지 않는다. 

x = x[0] + x[1]

from tensorflow.keras.utils import to_categorical
word_size = len(token.word_index)
print('사이즈 :',word_size)     #11



x = to_categorical(x)
print(x)
print(x.shape)    (1, 15, 11)
