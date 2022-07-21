import torch

import pandas as pd
import numpy as np

from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings(action='ignore')

from google.colab import drive
drive.mount('/content/drive')

!unzip '/content/drive/MyDrive/Dacon/신용카드 사기 거래 탐지 AI 경진대회/data/open.zip'

train_df = pd.read_csv('./train.csv') # Train
val_df = pd.read_csv('./val.csv') # Validation

val_normal, val_fraud = val_df['Class'].value_counts()
val_contamination = val_fraud / val_normal
print(f'Validation contamination : [{val_contamination}]')

# Train dataset은 Label이 존재하지 않음
train_x = train_df.drop(columns=['ID']) # Input Data

# 가설 설정 : Train dataset도 Validation dataset과 동일한 비율로 사기거래가 발생 했을 것이다. -> model parameter : contamination=val_contamination(=0.001055) 적용
model = EllipticEnvelope(support_fraction = 0.994, contamination = val_contamination, random_state = 42)
model.fit(train_x)

def get_pred_label(model, x, k):
  prob = model.score_samples(x)
  prob = torch.tensor(prob, dtype = torch.float)
  topk_indices = torch.topk(prob, k = k, largest = False).indices

  pred = torch.zeros(len(x), dtype = torch.long)
  pred[topk_indices] = 1
  return pred.tolist(), prob.tolist()

val_x = val_df.drop(columns=['ID', 'Class']) # Input Data
val_y = val_df['Class'] # Label

val_pred, val_prob = get_pred_label(model, val_x, 29)
val_score = f1_score(val_y, val_pred, average='macro')
print(f'Validation F1 Score : [{val_score}]')
print(classification_report(val_y, val_pred))
tn, fp, fn, tp = confusion_matrix(val_y, val_pred).ravel()
print('tp : ', tp, ', fp : ', fp, ', tn : ', tn, ', fn : ', fn)

test_df = pd.read_csv('./test.csv') # Train
test_df.head()

test_x = test_df.drop(columns=['ID'])

test_pred, _ = get_pred_label(model, test_x, 318)
print('n_fraud : ', sum(test_pred))

submit = pd.read_csv('./sample_submission.csv')
submit.head()

submit['Class'] = test_pred
submit.to_csv('/content/submit_EllipticEnvelope.csv', index=False)

pd.read_csv('/content/submit_EllipticEnvelope.csv')['Class'].sum()
