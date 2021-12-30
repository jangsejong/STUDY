from tensorflow.keras.preprocessing.text import Tokenizer

text = '나는 진짜 맛있는 밥을 진짜 너무 너무 맛있게 먹었다.'

token = Tokenizer()
token.fit_on_texts([text])

#print(token.word_index)  #{'진짜': 1, '너무': 2, '나는': 3, '맛있는': 4, '밥을': 5, '맛있게': 6, '먹었다': 7}

x = token.texts_to_sequences([text])
#print(x)   #[[3, 1, 4, 5, 1, 2, 2, 6, 7]]

from tensorflow.keras.utils import to_categorical
word_size = len(token.word_index)
print(word_size)     #7

x = to_categorical(x)
print(x, x.shape)    (1, 9, 8)
