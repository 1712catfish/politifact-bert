import tensorflow as tf
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import re
import nltk
import torch
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from gensim.utils import simple_preprocess
from math import ceil
from torch.optim import AdamW
from torcheval.metrics.functional import binary_auroc
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy, softmax
from torch.nn import CrossEntropyLoss
# from torcheval.metrics import *
from torchmetrics.classification import MulticlassAUROC
from sklearn.utils import gen_batches


class Project:

    def __init__(self):
        self.model_name = "bert-base-uncased"
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name, do_lower_case=True)
        self.max_seq_len = 50

        nltk.download("stopwords")
        self.stop_words = set(nltk.corpus.stopwords.words("english"))

        self.device = "cuda:0"

    def seq_preprocess(self, sequence):
        processed_words = simple_preprocess(
            re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(#[A-Za-z0-9]+)|(\w+:\/\/\S+)", " ", sequence))
        processed_words = [word.lower() for word in processed_words if word.lower() not in self.stop_words]

        return ' '.join(processed_words)

    def encode(self, texts, return_tensors='pt'):
        texts = [self.seq_preprocess(seq) for seq in texts]

        tokens = self.tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            max_length=self.max_seq_len,
            truncation=True,
            return_attention_mask=True,
            padding='max_length',
            return_tensors=return_tensors,
        )
        return {k: v.to(self.device) for k, v in tokens.items()}

    def prep_label(self, labels, categories=(True, False)):
        return torch.Tensor([
            [1 - categories.index(label), categories.index(label)] for label in labels
        ]).to(self.device)

    def accuracy(self, logits, labels):
        return torch.mean((logits > 0) == labels, dtype=torch.float32)

    def AUC(self, logits, labels):
        scores = []
        for i in range(logits.shape[-1]):
            preds = torch.as_tensor(logits[:, i] > 0, dtype=torch.float32)
            scores += [binary_auroc(preds, labels[:, 0])]

            # auc += [preds == labels[:, 0]]

        return torch.mean(torch.as_tensor(scores, dtype=torch.float32))



# self = Project()