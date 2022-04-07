import pandas as pd
import numpy as np

#csv 형식의 training 데이터를 로드합니다.
path = "D:\\Study\\_data\\dacon\\news\\"
train = pd.read_csv(path+ 'train.csv')
test = pd.read_csv(path +"test.csv") #파일 읽기
test.text = test.text.str.replace(r"\s+", " ", regex=True)
train.text = train.text.str.replace(r"\s+", " ", regex=True)

texts = list(train['text'])
target = list(train['target'])

texts_list = []
for text in texts:
  texts_list.append(text.split())

num_features = 300
min_word_count = 40
num_workers = 4
context = 10
downsampling = 1e-3

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim.models import word2vec
print("Training model ....")

model = word2vec.Word2Vec(texts_list, workers=num_workers, min_count=min_word_count, window=context, sample=downsampling)

# model_name = "20features_text"
# model.save(model_name)

def get_features(words, model, num_features):
      # 출력 벡터 초기화
  feature_vector = np.zeros((num_features), dtype=np.float32)

  num_words = 0
  # 어휘사전 준비
  index2word_set = set(model.wv.index2word)

  for w in words:
    if w in index2word_set:
      num_words +=1
      # 사전에 해당하는 단어에 대해 단어 벡터를 더함
      feature_vector = np.add(feature_vector, model[w])

  # 문장의 단어 수만큼 나누어 단어 벡터의 평균값을 문장 벡터로 함
  feature_vector = np.divide(feature_vector, num_words)
  return feature_vector

def get_dataset(reviews, model, num_features):
      dataset = list()

  for s in texts:
    dataset.append(get_features(s, model, num_features))

  textFeatureVecs = np.stack(dataset)

  return textFeatureVecs

train_data_vecs = get_dataset(sentences, model, num_features)

from sklearn.model_selection import train_test_split
import numpy as np

X = train_data_vecs
y = np.array(sentiments)

RANDOM_SEED = 42
TEST_SPLIT = 0.2

X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=TEST_SPLIT, random_state=RANDOM_SEED)

from sklearn.linear_model import LogisticRegression

lgs = LogisticRegression(class_weight='balanced')
lgs.fit(X_train, y_train)



# X = train.text #training 데이터에서 문서 추출
# y = train.target #training 데이터에서 라벨 추출
# X.head() #데이터 살펴보기
# y.head() #데이터 살펴보기



# # from sklearn.feature_extraction.text import CountVectorizer #sklearn 패키지의 CountVectorizer import
# from sklearn.feature_extraction.text import TfidfVectorizer

# sample_vectorizer = TfidfVectorizer(analyzer ='char_wb') #객체 생성

# sample_text1 = ["hello, my name is dacon and I am a data scientist!"]

# sample_vectorizer.fit(sample_text1) # 

# print(sample_vectorizer.vocabulary_) #

# sample_text2 = ["you are learning dacon data science"]

# sample_vector = sample_vectorizer.transform(sample_text2)
# print(sample_vector.toarray())

# sample_text3 = ["you are learning dacon data science with news data"]

# sample_vector2 = sample_vectorizer.transform(sample_text3)
# print(sample_vector2.toarray())




# vectorizer = TfidfVectorizer() #TfidfVectorizer 객체 생성
# vectorizer.fit(X) #  학습
# X = vectorizer.transform(X) #transform

# vectorizer.inverse_transform(X[0]) #역변환하여 첫번째 문장의 단어들 확인

# from sklearn.linear_model import LogisticRegression #모델 불러오기
# model = LogisticRegression( max_iter=250, random_state= 66) #객체에 모델 할당
# model.fit(X, y) #모델 학습

# from sklearn.metrics import accuracy_score

# #run model
# y_pred = model.predict(X)
# # print('예측 라벨 : ', y_pred)
# # print('실제 라벨 : ', train.target[0])

# from sklearn.metrics import accuracy_score
# acc = accuracy_score(y, y_pred)
# print('accuracy : ', acc)


# test_X = test.text #문서 데이터 생성
# # test_y = test.target #라벨 데이터 생성

# test_X_vect = vectorizer.transform(test_X) #문서 데이터 transform 
# #test 데이터를 대상으로 fit_transform 메소드를 실행하는 것은 test 데이터를 활용해 vectorizer 를 학습 시키는 것으롤 data leakage 에 해당합니다.

# pred = model.predict(test_X_vect) #test 데이터 예측
# print(pred)

# submission = pd.read_csv(path +"sample_submission.csv") #제출용 파일 불러오기
# submission.head() #제출 파일이 잘 생성되었는지 확인

# submission["target"] = pred #예측 값 넣어주기
# submission.head() # 데이터가 잘 들어갔는지 확인합니다.

# # submission을 csv 파일로 저장합니다.
# # index=False란 추가적인 id를 부여할 필요가 없다는 뜻입니다. 
# # 정확한 채점을 위해 꼭 index=False를 넣어주세요.
# submission.to_csv(path +"0406_07.csv",index=False)

