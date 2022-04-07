import pandas as pd


#csv 형식의 training 데이터를 로드합니다.
path = "D:\\Study\\_data\\dacon\\news\\"
train_df  = pd.read_csv(path+ 'train.csv')
test_df  = pd.read_csv(path +"test.csv") #파일 읽기

train_df["text"]  = train_df["text"].str.replace("\s+", " ", regex=True)
test_df["text"]  = test_df["text"].str.replace("\s+", " ", regex=True)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


def get_pipe(model, model_name: str) -> Pipeline:
    "TfidfVectorizer와 모델을 연결한 파이프라인을 반환하는 함수"
    tfidf = TfidfVectorizer(analyzer="char", ngram_range=(1, 3))
    pipe = Pipeline([
        ("tfidf", tfidf),
        (model_name, model)
    ])
    return pipe

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

def return_kfold_accuarcy(model, k: int = 5) -> float:
    "모델을 입력받아 KFold 예측 후 accuracy score를 반환하는 함수"
    kfold = StratifiedKFold(k, shuffle=True, random_state=42)
    result = []
    for train_idx, test_idx in kfold.split(train_df["text"], train_df["target"]):
        train, val = train_df.iloc[train_idx], train_df.iloc[test_idx]
        model.fit(train["text"], train["target"])
        pred = model.predict(val["text"])
        acc = accuracy_score(val["target"], pred)
        result.append(acc)

    return np.mean(result)

from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC


models = [
    ("naive_bayes", BernoulliNB()),
    ("SGD", SGDClassifier(random_state=42, n_jobs=-1)),
    ("rfc", RandomForestClassifier(random_state=42, n_jobs=-1)),
    ("SVC", SVC(random_state=42)),
    ("ada", AdaBoostClassifier(random_state=42))
]

model_pipes = [(name, get_pipe(model, name)) for name, model in models]

from tqdm.auto import tqdm  # 진행바 라이브러리

# table = title="Model Comparison Table"

# for model_name, model in tqdm(model_pipes, leave=False):
#     acc = return_kfold_accuarcy(model)
#     table.add_row(model_name, f"{acc:0.3f}")


from sklearn.ensemble import StackingClassifier

stack_models = [(name, get_pipe(model, name)) for name, model in models]

stacking = StackingClassifier(stack_models)
acc = return_kfold_accuarcy(stacking)
print(acc)

stacking.fit(train_df["text"], train_df["target"])
submission_pred = stacking.predict(test_df["text"])

submission = pd.read_csv(path + "0406.csv")
submission["target"] = submission_pred
submission

