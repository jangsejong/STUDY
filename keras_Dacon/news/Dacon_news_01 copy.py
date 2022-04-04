import pandas as pd

#csv 형식의 training 데이터를 로드합니다.
path = "D:\\Study\\_data\\dacon\\news\\"
train = pd.read_csv(path+ 'train.csv')

#데이터 살펴보기 위해 데이터 최상단의 5줄을 표시합니다.
train.head() 
print(train.shape)

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

from sklearn.feature_extraction.text import CountVectorizer #sklearn 패키지의 CountVectorizer import

sample_vectorizer = CountVectorizer() #객체 생성

sample_text1 = ["hello, my name is dacon and I am a data scientist!"]

sample_vectorizer.fit(sample_text1) #CountVectorizer 학습

print(sample_vectorizer.vocabulary_) #Vocabulary

sample_text2 = ["you are learning dacon data science"]

sample_vector = sample_vectorizer.transform(sample_text2)
print(sample_vector.toarray())

sample_text3 = ["you are learning dacon data science with news data"]

sample_vector2 = sample_vectorizer.transform(sample_text3)
print(sample_vector2.toarray())




vectorizer = CountVectorizer() #countvectorizer 생성
vectorizer.fit(X) #countvectorizer 학습
X = vectorizer.transform(X) #transform

vectorizer.inverse_transform(X[0]) #역변환하여 첫번째 문장의 단어들 확인

from xgboost import XGBRegressor, XGBClassifier

model = XGBRegressor(max_iter=500) #객체에 모델 할당
model.fit(X, y) #모델 학습

from sklearn.metrics import accuracy_score

#run model
y_pred = model.predict(X[0])
print('예측 라벨 : ', y_pred)
print('실제 라벨 : ', train.target[0])


test = pd.read_csv(path +"test.csv") #파일 읽기
test.head() #파일 확인

test_X = test.text #문서 데이터 생성

test_X_vect = vectorizer.transform(test_X) #문서 데이터 transform 
#test 데이터를 대상으로 fit_transform 메소드를 실행하는 것은 test 데이터를 활용해 vectorizer 를 학습 시키는 것으롤 data leakage 에 해당합니다.

pred = model.predict(test_X_vect) #test 데이터 예측
pred = pred.astype(int)
print(pred)

submission = pd.read_csv(path +"sample_submission.csv") #제출용 파일 불러오기
submission.head() #제출 파일이 잘 생성되었는지 확인

submission["target"] = pred #예측 값 넣어주기
submission.head() # 데이터가 잘 들어갔는지 확인합니다.

# submission을 csv 파일로 저장합니다.
# index=False란 추가적인 id를 부여할 필요가 없다는 뜻입니다. 
# 정확한 채점을 위해 꼭 index=False를 넣어주세요.
submission.to_csv(path +"0404_03.csv",index=False)


