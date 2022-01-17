# 원-핫 인코딩 (One-Hot Encoding)
#link : https://wikidocs.net/22647
'''
from konlpy.tag import Okt  
okt = Okt()  
token = okt.morphs("나는 자연어 처리를 배운다")  
print(token)

word2index = {}
for voca in token:
  if voca not in word2index.keys():
    word2index[voca] = len(word2index)
print(word2index)

def one_hot_encoding(word, word2index):
      one_hot_vector = [0]*(len(word2index))
  index = word2index[word]
  one_hot_vector[index] = 1
  return one_hot_vector

one_hot_encoding("자연어", word2index)
'''

text = "나랑 점심 먹으러 갈래 점심 메뉴는 햄버거 갈래 갈래 햄버거 최고야"
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

text = "나랑 점심 먹으러 갈래 점심 메뉴는 햄버거 갈래 갈래 햄버거 최고야"

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
print(tokenizer.word_index) # 각 단어에 대한 인코딩 결과 출력.

sub_text = "점심 먹으러 갈래 메뉴는 햄버거 최고야"
encoded = tokenizer.texts_to_sequences([sub_text])[0]
print(encoded)

one_hot = to_categorical(encoded)
print(one_hot)