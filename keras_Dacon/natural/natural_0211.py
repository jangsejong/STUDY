import numpy as np
import pandas as pd

from transformers import pipeline, AutoTokenizer
from tqdm import tqdm

path = 'D:\\Study\\_data\\dacon\\natural\\'

# tokenizer = AutoTokenizer.from_pretrained("Huffon/qnli-model")
# model = pipeline("text-classification", model="Huffon/qnli-model", return_all_scores=False)

# tokenizer = AutoTokenizer.from_pretrained("Huffon/klue-roberta-base-nli")
# model = pipeline("text-classification", model="Huffon/klue-roberta-base-nli", return_all_scores=False)

# tokenizer = AutoTokenizer.from_pretrained("Huffon/sentiment-analysis-roberta-base")
# model = pipeline("text-classification", model="Huffon/sentiment-analysis-roberta-base", return_all_scores=False)

tokenizer = AutoTokenizer.from_pretrained("Doogie/wav2vec2-korea-doogie-test-01")
model = pipeline("text-classification", model="Doogie/wav2vec2-korea-doogie-test-01", return_all_scores=False)

df_test = pd.read_csv(path +"test_data.csv")
X_test = df_test["premise"] + " " + tokenizer.sep_token + " " + df_test["hypothesis"]

y_preds = []
for i in tqdm(range(X_test.shape[0])):
    y_pred = model(X_test[i])[0]["label"].lower()
    y_preds.append(y_pred)

y_preds = np.array(y_preds)

df_test["label"] = y_preds
df_submission = df_test.loc[:, ["index", "label"]]
df_submission.to_csv(path+ "natural_0211_01.csv", index=False)
















