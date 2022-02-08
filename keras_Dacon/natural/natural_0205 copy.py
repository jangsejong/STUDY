import numpy as np
import pandas as pd

from transformers import pipeline, AutoTokenizer
from tqdm import tqdm

path = 'D:\\Study\\_data\\dacon\\natural\\'

predefined_args = {
        'attention_cell': 'multi_head',
        'num_layers': 12,
        'units': 768,
        'hidden_size': 3072,
        'max_length': 512,
        'num_heads': 12,
        'scaled': True,
        'dropout': 0.1,
        'use_residual': True,
        'embed_size': 768,
        'embed_dropout': 0.1,
        'token_type_vocab_size': 2,
        'word_embed': None,
    }
path = 'D:\\Study\\_data\\dacon\\natural\\'

train = pd.read_csv(path+"train_data.csv")
test  = pd.read_csv(path+"test_data.csv")
sub   = pd.read_csv(path+"sample_submission.csv")
# from gluonnlp.data import SentencepieceTokenizer
# from kobert import get_tokenizer
# import onnxruntime
# from kobert import get_onnx_kobert_model

# tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-small-v2-discriminator")
# model = pipeline("text-classification", model="monologg/koelectra-small-v2-discriminator", return_all_scores=False)
tokenizer = AutoTokenizer.from_pretrained(
   "monologg/koelectra-small-v2-discriminator"
)
# model = ElectraForSequenceClassification.from_pretrained("monologg/koelectra-small-v2-discriminator")
model = pipeline("text-classification", model="monologg/koelectra-small-v2-discriminator", return_all_scores=False)

df_test = pd.read_csv(path +"test_data.csv")
X_test = df_test["premise"] + " " + tokenizer.sep_token + " " + df_test["hypothesis"]

y_preds = []
for i in tqdm(range(X_test.shape[0])):
    y_pred = model(X_test[i])[0]["label"].lower()
    y_preds.append(y_pred)

y_preds = np.array(y_preds)

df_test["label"] = y_preds
df_submission = df_test.loc[:, ["index", "label"]]
df_submission.to_csv(path+ "natural_0205_7942.csv", index=False)
















