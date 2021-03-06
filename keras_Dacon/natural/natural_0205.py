import numpy as np
import pandas as pd

from transformers import pipeline, AutoTokenizer
from tqdm import tqdm

path = 'D:\\Study\\_data\\dacon\\natural\\'

tokenizer = AutoTokenizer.from_pretrained("hyyoka/wav2vec2-xlsr-korean-senior")
model = pipeline("text-classification", model="hyyoka/wav2vec2-xlsr-korean-senior", return_all_scores=False)

# tokenizer = AutoTokenizer.from_pretrained("Huffon/klue-roberta-base-nli")
# model = pipeline("text-classification", model="Huffon/klue-roberta-base-nli", return_all_scores=False)

# tokenizer = AutoTokenizer.from_pretrained("beomi/korean-hatespeech-multilabel")
# model = pipeline("text-classification", model="beomi/korean-hatespeech-multilabel", return_all_scores=False)

# tokenizer = AutoTokenizer.from_pretrained("Huffon/klue-roberta-pretrain-nli")
# model = pipeline("text-classification", model="Huffon/klue-roberta-pretrain-nli", return_all_scores=False)

df_test = pd.read_csv(path +"test_data.csv")
X_test = df_test["premise"] + " " + tokenizer.sep_token + " " + df_test["hypothesis"]

y_preds = []
for i in tqdm(range(X_test.shape[0])):
    y_pred = model(X_test[i])[0]["label"].lower()
    y_preds.append(y_pred)

y_preds = np.array(y_preds)

df_test["label"] = y_preds
df_submission = df_test.loc[:, ["index", "label"]]
df_submission.to_csv(path+ "natural_0214_01.csv", index=False)

'''
beomi/korean-hatespeech-multilabel

'''
















