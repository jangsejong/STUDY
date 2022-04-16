import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyrsistent import v
import seaborn as sns

import warnings
warnings.filterwarnings(action='ignore')

from sklearn.metrics import accuracy_score

import nltk  
from nltk import tokenize
from nltk.corpus import names, stopwords, words
from nltk.stem import WordNetLemmatizer

import re

from wordcloud import WordCloud

path = "D:\\Study\\_data\\dacon\\news\\"
train = pd.read_csv(path+ 'train.csv', index_col='id')
test = pd.read_csv(path +"test.csv", index_col='id') #파일 읽기

print(train.shape)
train.head(3)

# test = pd.read_csv("test.csv", index_col='id')

print(test.shape)
test.head(3)
submission = pd.read_csv(path +"sample_submission.csv", index_col='id') #제출용 파일 불러오기
# submission = pd.read_csv("sample_submission.csv", index_col='id')

print(submission.shape)
submission.head(2)

### 뉴스분류에 대한 각종 설명이 필요할 경우를 대비해서 dict 형으로 선언했습니다 

dict_label = {1 : "alt.atheism",
              2 : "comp.graphics",
              3 : "comp.os.ms-windows.misc",
              4 : "comp.sys.ibm.pc.hardware",
              5 : "comp.sys.mac.hardware",
              6 : "comp.windows.x",
              7 : "misc.forsale",
              8 : "rec.autos",
              9 : "rec.motorcycles",
              10 : "rec.sport.baseball",
              11 : "rec.sport.hockey",
              12 : "sci.crypt",
              13 : "sci.electronics",
              14 : "sci.med",
              15 : "sci.space",
              16 : "soc.religion.christian",
              17 : "talk.politics.guns",        
              18 : "talk.politics.mideast",
              19 : "talk.politics.misc",
              20 : "talk.religion.misc"}



train.info()
train.isna().sum()

# 결측치 존재 여부를 확인해주는 함수
def check_missing_col(dataframe):
    missing_col = []
    counted_missing_col = 0
    for i, col in enumerate(dataframe.columns):
        missing_values = sum(dataframe[col].isna())
        is_missing = True if missing_values >= 1 else False
        if is_missing:
            counted_missing_col += 1
            print(f'결측치가 있는 컬럼은: {col}입니다')
            print(f'해당 컬럼에 총 {missing_values}개의 결측치가 존재합니다.')
            missing_col.append([col, dataframe[col].dtype])
    if counted_missing_col == 0:
        print('결측치가 존재하지 않습니다')
    return missing_col

missing_col = check_missing_col(train)

def reg_func1(text):
    return re.sub("[\n]", " ", text)

def reg_func2(text):
    return re.sub("[^0-9a-zA-Z. ]", "", text)

### Test : 적용 전 :
train.loc[9232,"text"]

### Test : 적용 후,  불필요한 " "(중간)+(문장 앞/뒤) 여백제거 + 정규표현식 함수1, 2 적용

train["text"].str.replace("\s+", " ", regex=True).str.strip().apply(reg_func1).apply(reg_func2)


train["text_re"] = train["text"].str.replace("\s+", " ", regex=True).str.strip().apply(reg_func1).apply(reg_func2)

print(train.shape)
train.loc[6,"text_re"]

#### for Test dataset :

test["text_re"] = test["text"].str.replace("\s+", " ", regex=True).str.strip().apply(reg_func1).apply(reg_func2)

print(test.shape)
test.loc[6,"text_re"]

## Corpus 말뭉치 참고용 : 상세한 내용은 위에 소개한 NLTK 링크에서 확인해보세요

names.fileids()      # 남자, 여자 영문이름 말뭉치
names.words("female.txt")
words.fileids()      # 영어사전 말뭉치
words.words('en')
stopwords.fileids()  # 불용어사전 말뭉치
stopwords.words("english")

### Eng stopword dictionary 
stopwords.fileids() 
stopwords.words('english')

STOPWORDS = stopwords.words('english')
### Add by customized :
STOPWORDS.append("one")

### 불용어(stopwords) 처리용 함수 생성 :

def del_stopwords(text):
    output = []
    
    for word in tokenize.word_tokenize(text.lower()):
        if word not in STOPWORDS:
            output.append(word)
    
    return " ".join(output)

train.loc[6,"text_re"]
del_stopwords(train.loc[6,"text_re"])

train["text_sw"] = train["text_re"].apply(del_stopwords)

print(train.shape)
train.loc[6,"text_sw"]

#### for Test dataset :

test["text_sw"] = test["text_re"].apply(del_stopwords)

print(test.shape)
test.loc[6,"text_sw"]

### 품사 tag 기준으로 걸러낼 경우 : 

def filter_pos_tag(text):
    lst_tag = ['CD','JJ','LS','NN','NNP','NNPS','NNS','RB','VB','VBD','VBG','VBN','VBP','VBZ']
    output = []
    
    for word, pos in nltk.pos_tag(tokenize.word_tokenize(text, language='english')):
        if pos in lst_tag:
            output.append(word)
    
    return " ".join(output)
### 품사 tag 를 넘어서, 기본형(동사/명사/형용사/부사) 으로 걸러낼 경우 :

def filter_pos_tag_lemmat(text):
    lst_tag_a = ['JJ','JJR','JJS']
    lst_tag_n = ['NN','NNP','NNPS','NNS']
    lst_tag_r = ['RB','RBR','RBS']
    lst_tag_v = ['VB','VBD','VBG','VBN','VBP','VBZ']
    output = []
    wnl = WordNetLemmatizer()
    
    for word, pos in nltk.pos_tag(tokenize.word_tokenize(text, language='english')):
        if pos in lst_tag_a:
            word = wnl.lemmatize(word, 'a')
        elif pos in lst_tag_n:
            word = wnl.lemmatize(word, 'n')
        elif pos in lst_tag_r:
            word = wnl.lemmatize(word, 'r')
        elif pos in lst_tag_v:
            word = wnl.lemmatize(word, 'v')

        output.append(word)
    
    return " ".join(output)
train["text_pos"] = train["text_sw"].apply(filter_pos_tag)

print(train.shape)
train.loc[6,"text_pos"]

#### for Test dataset 

test["text_pos"] = test["text_sw"].apply(filter_pos_tag)

print(test.shape)
test.loc[6,"text_pos"]

train["text_lemm"] = train["text_sw"].apply(filter_pos_tag_lemmat)

print(train.shape)
train.loc[6,"text_lemm"]
#### for Test dataset 

test["text_lemm"] = test["text_sw"].apply(filter_pos_tag_lemmat)

print(test.shape)
test.loc[6,"text_lemm"]

train["length"] = train["text_lemm"].apply(lambda x : len(x))

print(train.shape)
train.head(1)

#### for Test dataset : 

test["length"] = test["text_lemm"].apply(lambda x : len(x))

print(test.shape)
test.head(1)

train["target"].nunique()
train["target"].value_counts().sort_index().index
train["target"].value_counts().sort_index().values 
train["target"].value_counts(normalize=True).sort_index().values

train.groupby("target").mean().reset_index()

plt.figure(figsize=(12,4))
sns.countplot(train["target"])
#### Texting
for t in range(train["target"].nunique()):
    x_pos = train["target"].value_counts().sort_index().index[t]
    y_pos = train["target"].value_counts().sort_index().values[t] 
    y_rate = train["target"].value_counts(normalize=True).sort_index().values[t]
    y_len = train.loc[train["target"] == t, 'length'].mean()
    plt.text(x_pos, y_pos, str(y_pos), size=10, ha="center")
    plt.text(x_pos, y_pos-50, str(f"{y_rate:.3f}"), size=8, ha="center", va='bottom')
    plt.text(x_pos, y_pos/150, str(f"{y_len:.1f}"), size=8, ha="center")
#     plt.text(x_pos, 100+50*(x_pos%2), str(f"{y_len:.1f}"), size=8, ha="center")

plt.title("Target w/ counts, rates and length")    
plt.show()

plt.figure(figsize=(12,4))
sns.countplot(train["target"])
sns.pointplot(x=train.groupby("target").mean().reset_index()['target'], y=train.groupby("target").mean().reset_index()["length"])
plt.legend(dict_label.items(), loc = 'lower right', fontsize=8)
plt.title("Target with length trend")    

train.loc[train["target"] == 2, "text_pos"]
train.loc[46, "text_pos"]
train.loc[train["target"] == 2, ["text","text_re","text_sw","text_pos","text_lemm","length"]].sort_values(by='length')