import pandas as pd

#csv 형식의 training 데이터를 로드합니다.
path = "D:\\Study\\_data\\dacon\\news\\"
train = pd.read_csv(path+ 'train.csv')
test = pd.read_csv(path +"test.csv") #파일 읽기
test.text = test.text.str.replace(r"\s+", " ", regex=True)
train.text = train.text.str.replace(r"\s+", " ", regex=True)

#데이터 살펴보기 위해 데이터 최상단의 5줄을 표시합니다.
# train.head() 
# print(train.shape)

# import re
# import string
# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords  
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences

# train["length"] = train.text.map(len)
# test["length"] = test.text.map(len)

# STOP_WORDS = stopwords.words('english')
# print("stop words", STOP_WORDS)

# def remove_stop_words(s):
#       return " ".join([x for x in word_tokenize(s)])

# def preprocess_data(train, test, 
#                     no_number=False,
#                     no_stopwords=False,
#                     no_punctuation=False,
#                     min_len=0,
#                     lowercase=False):
#   train, test = train.copy(), test.copy()

#   for df in [train, test]:
#     # 띄어쓰기나 공백이 연속된 경우 공백 하나로 바꿈
#     df.text = df.text.str.replace(r"\s+", " ", regex=True)

#     if lowercase: # 소문자로 변경
#       df.text = df.text.str.lower()
#     if no_number: # 숫자 제거
#       df.text = df.text.str.replace(r"\d+", "", regex=True)
#     if no_punctuation: # punctuation 제거
#       df.text = df.text.str.translate(str.maketrans('', '', string.punctuation))
#     if no_stopwords: # 불용어 제거
#       df.text = df.text.map(remove_stop_words)
    
#     df["length"] = df.text.map(len)
    
#     # 길이가 min_len 미만인 문자열은 학습 데이터에서 제거한다
#     if min_len > 0 and "target" in df.columns:
#       df.drop(df[df.length < min_len].index, inplace=True)

#   return train, test


# def tokenize_data(train, test, vocab_size, max_len):
#   """
#     Keras Tokenizer를 이용해 토큰화한 뒤 pad_sequences를 이용해 패딩을 추가함.
#   """
#   tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
#   tokenizer.fit_on_texts(train.text)

#   for df in [train, test]:
#     df["encoded"] = pad_sequences(
#          tokenizer.texts_to_sequences(df.text),
#          maxlen=max_len,
#          padding="post"
#          ).tolist()

# vocab_size = 10000
# max_len = 512
# preprocess_params = {
#   "no_number" : True,
#   "no_stopwords" : True,
#   "no_punctuation" : True,
#   "lowercase" : True,
#   "min_len" : 30
# }
# tokenization_params = {
#     "vocab_size": vocab_size,
#     "max_len": max_len
# }

# train, test = preprocess_data(
#     train, 
#     test,
#     **preprocess_params
#     )
# tokenize_data(train, test, **tokenization_params)



def check_missing_col(dataframe):
    missing_col = []
    for col in dataframe.columns:
        missing_values = sum(dataframe[col].isna())
        is_missing = True if missing_values >= 1 else False
        if is_missing:
            print(f'결측치가 있는 컬럼은: {col} 입니다')
            print(f'해당 컬럼에 총 {missing_values} 개의 결측치가 존재합니다.')
            missing_col.append([col, dataframe[col].dtype])
    if missing_col == []:
        print('결측치가 존재하지 않습니다')
    return missing_col

missing_col = check_missing_col(train)

X = train.text #training 데이터에서 문서 추출
y = train.target #training 데이터에서 라벨 추출
X.head() #데이터 살펴보기
y.head() #데이터 살펴보기



# from sklearn.feature_extraction.text import CountVectorizer #sklearn 패키지의 CountVectorizer import
from sklearn.feature_extraction.text import TfidfVectorizer

sample_vectorizer = TfidfVectorizer(analyzer ='char_wb') #객체 생성

sample_text1 = ["hello, my name is dacon and I am a data scientist!"]

sample_vectorizer.fit(sample_text1) # 

print(sample_vectorizer.vocabulary_) #

sample_text2 = ["you are learning dacon data science"]

sample_vector = sample_vectorizer.transform(sample_text2)
print(sample_vector.toarray())

sample_text3 = ["you are learning dacon data science with news data"]

sample_vector2 = sample_vectorizer.transform(sample_text3)
print(sample_vector2.toarray())




vectorizer = TfidfVectorizer() #TfidfVectorizer 객체 생성
vectorizer.fit(X) #  학습
X = vectorizer.transform(X) #transform

vectorizer.inverse_transform(X[0]) #역변환하여 첫번째 문장의 단어들 확인

from sklearn.linear_model import LogisticRegression #모델 불러오기
model = LogisticRegression( max_iter=250, random_state= 66) #객체에 모델 할당
model.fit(X, y) #모델 학습

from sklearn.metrics import accuracy_score

#run model
y_pred = model.predict(X)
# print('예측 라벨 : ', y_pred)
# print('실제 라벨 : ', train.target[0])

from sklearn.metrics import accuracy_score
acc = accuracy_score(y, y_pred)
print('accuracy : ', acc)


test_X = test.text #문서 데이터 생성
# test_y = test.target #라벨 데이터 생성

test_X_vect = vectorizer.transform(test_X) #문서 데이터 transform 
#test 데이터를 대상으로 fit_transform 메소드를 실행하는 것은 test 데이터를 활용해 vectorizer 를 학습 시키는 것으롤 data leakage 에 해당합니다.

pred = model.predict(test_X_vect) #test 데이터 예측
print(pred)

submission = pd.read_csv(path +"sample_submission.csv") #제출용 파일 불러오기
submission.head() #제출 파일이 잘 생성되었는지 확인

submission["target"] = pred #예측 값 넣어주기
submission.head() # 데이터가 잘 들어갔는지 확인합니다.

# submission을 csv 파일로 저장합니다.
# index=False란 추가적인 id를 부여할 필요가 없다는 뜻입니다. 
# 정확한 채점을 위해 꼭 index=False를 넣어주세요.
submission.to_csv(path +"0406_07.csv",index=False)

