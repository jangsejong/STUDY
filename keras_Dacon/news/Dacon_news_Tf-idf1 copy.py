import pandas as pd

#csv 형식의 training 데이터를 로드합니다.
path = "D:\\Study\\_data\\dacon\\news\\"
train = pd.read_csv(path+ 'train.csv')
test = pd.read_csv(path +"test.csv") #파일 읽기
test.text = test.text.str.replace(r"\s+", " ", regex=True)
train.text = train.text.str.replace(r"\s+", " ", regex=True)



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
test_email = test.data
test_label = test.target

vocab_size = 10000
num_classes = 20

from tensorflow.keras.preprocessing.text import Tokenizer

def prepare_data(train_data, test_data, mode): # 전처리 함수
    tokenizer = Tokenizer(num_words = vocab_size) # vocab_size 개수만큼의 단어만 사용한다.
    tokenizer.fit_on_texts(train_data)
    X_train = tokenizer.texts_to_matrix(train_data, mode=mode) # 샘플 수 × vocab_size 크기의 행렬 생성
    X_test = tokenizer.texts_to_matrix(test_data, mode=mode) # 샘플 수 × vocab_size 크기의 행렬 생성
    return X_train, X_test, tokenizer.index_word

from tensorflow.keras.utils import to_categorical

X_train, X_test, index_to_word = prepare_data(X, test_email, 'binary') # binary 모드로 변환
y_train = to_categorical(y, num_classes) # 원-핫 인코딩
y_test = to_categorical(test_label, num_classes) # 원-핫 인코딩







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

import numpy as np


vectorizer = TfidfVectorizer()

vectorizer.fit(X)

train_vec = vectorizer.transform(X)
train_y = y

test_vec = vectorizer.transform(test_email)




from sklearn.neural_network import MLPClassifier
model = MLPClassifier(activation='tanh', solver='adam', alpha=0.0001, hidden_layer_sizes=(100,), random_state=66)
model.fit(train_vec, train_y)




#run model
y_pred = model.predict(test_vec)
# print('예측 라벨 : ', y_pred)
# print('실제 라벨 : ', train.target[0])

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
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
submission.to_csv(path +"0408_01.csv",index=False)

