
import os
from zipfile import ZipFile
import torch
from transformers import BertModel
import gluonnlp as nlp

from kobert import download, get_tokenizer


def get_pytorch_kobert_model(ctx="cpu", cachedir=".cache"):
    def get_kobert_model(model_path, vocab_file, ctx="cpu"):
        bertmodel = BertModel.from_pretrained(model_path, return_dict=False)
        device = torch.device(ctx)
        bertmodel.to(device)
        bertmodel.eval()
        vocab_b_obj = nlp.vocab.BERTVocab.from_sentencepiece(
            vocab_file, padding_token="[PAD]"
        )
        return bertmodel, vocab_b_obj

    pytorch_kobert = {
        "url": "s3://skt-lsl-nlp-model/KoBERT/models/kobert_v1.zip",
        "chksum": "411b242919",  # 411b2429199bc04558576acdcac6d498
    }

    # download model
    model_info = pytorch_kobert
    model_path, is_cached = download(
        model_info["url"], model_info["chksum"], cachedir=cachedir
    )
    cachedir_full = os.path.expanduser(cachedir)
    zipf = ZipFile(os.path.expanduser(model_path))
    zipf.extractall(path=cachedir_full)
    model_path = os.path.join(os.path.expanduser(cachedir), "kobert_from_pretrained")
    # download vocab
    vocab_path = get_tokenizer()
    return get_kobert_model(model_path, vocab_path, ctx)


if __name__ == "__main__":
    import torch
    from kobert import get_pytorch_kobert_model

    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    model, vocab = get_pytorch_kobert_model()
    sequence_output, pooled_output = model(input_ids, input_mask, token_type_ids)
    print(pooled_output.shape)
    print(vocab)
    print(sequence_output[0])