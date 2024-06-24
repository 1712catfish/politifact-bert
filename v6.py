import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

from sklearn.utils import gen_batches
import sklearn
import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import nltk
import torch
from sklearn.metrics import *
from transformers import BertTokenizer, BertForSequenceClassification
from gensim.utils import simple_preprocess
from torch.optim import AdamW
from torcheval.metrics.functional import binary_auroc
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import gen_batches
from sklearn.metrics import roc_auc_score
from torch_geometric.nn.conv import GATConv, GATv2Conv
from torch_geometric.utils import mask_to_index, dense_to_sparse, select
from prefetch_generator import background
import nltk
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as naf

from nlpaug.util import Action
import random


def punct_insertion(sentence, p=0.3, punctuations=None):
    if punctuations is None:
        punctuations = ['.', ';', '?', ':', '!', ',']
    sentence = sentence.strip().split(' ')
    len_sentence = len(sentence)

    num_punctuations = random.randint(1, int(len_sentence * p))
    augmented_sentence = sentence.copy()

    for _ in range(num_punctuations):
        punct = random.choice(punctuations)
        pos = random.randint(0, len(augmented_sentence) - 1)
        augmented_sentence = augmented_sentence[:pos] + [punct] + augmented_sentence[pos:]
    augmented_sentence = ' '.join(augmented_sentence)

    return augmented_sentence


def segment_shuffle(sentences, aug_max=10):
    sentences = [sentence.split(' ') for sentence in sentences]

    snips = [""] * len(sentences)
    for i, s in enumerate(sentences):
        pos1 = random.randint(0, len(s))
        pos2 = random.randint(0, len(s))

        if pos1 > pos2:
            pos1, pos2 = pos2, pos1
        if pos2 - pos1 > aug_max:
            pos1, pos2 = (pos1 + pos2 - aug_max) // 2, (pos1 + pos2 + aug_max) // 2
        snips[i] = s[pos1:pos2]
        sentences[i] = s[:pos1] + s[pos2:]

    snips = sklearn.utils.shuffle(snips)

    for i, snip in enumerate(snips):
        s = sentences[i]
        pos = random.choice((0, len(s)))
        sentences[i] = s[:pos] + snip + s[pos:]

    return [' '.join(s) for s in sentences]


class DataMixin:
    def __init__(self):
        self.cap = -1
        self.batch_size = 16
        self.data = None
        self.max_seq_len = 200
        self.device = "cuda:0"
        self.tokenizer = None
        self.model_name = "bert-base-uncased"

        nltk.download('stopwords')
        nltk.download('wordnet')
        os.system("unzip /usr/share/nltk_data/corpora/wordnet.zip -d /usr/share/nltk_data/corpora/")

        self.train_csv_path = 'politifact_train.csv'
        self.test_csv_path = 'politifact_test.csv'

        self.data = {
            'train_pandas': pd.read_csv(self.train_csv_path),
            'test_pandas': pd.read_csv(self.test_csv_path)
        }
        self.data['train_slices'] = self.get_slices(self.data['train_pandas'])
        self.data['test_slices'] = self.get_slices(self.data['test_pandas'])

        self.tokenizer = BertTokenizer.from_pretrained(self.model_name, do_lower_case=True)

        self.tta = False
        self.aug = naf.Sometimes([
            naw.SynonymAug(aug_src='wordnet'),
            naw.RandomWordAug(action="crop"),
            naw.RandomWordAug(action="swap"),
        ])

        self.is_train = False

    def tokenize(self, t):
        if self.tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name, do_lower_case=True)

        return self.tokenizer.batch_encode_plus(
            t,
            add_special_tokens=True,
            max_length=self.max_seq_len,
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            padding='max_length',
            return_tensors='pt',
        )

    def tokenize2(self, t1, t2):
        return self.tokenizer.batch_encode_plus(
            zip(t1, t2),
            add_special_tokens=True,
            max_length=self.max_seq_len,
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            padding='max_length',
            return_tensors='pt',
        )

    def get_slices(self, df):
        x, a, b = df['claim'], df['evidence'], df['label']

        indices = [i for i, (x1, x2) in enumerate(zip(x[:-1], x[1:])) if x1 != x2]
        indices = [0] + indices + [len(x)]

        slices = [slice(x, y, 1) for x, y in zip(indices[:-1], indices[1:])]

        return slices

    def get_aug(self, text_batch):
        text_batch = segment_shuffle(text_batch)
        text_batch = [punct_insertion(text) for text in text_batch]
        text_batch = self.aug.augment(text_batch)
        return text_batch

    def load_fn(self, df):
        t1, t2 = df['claim'].values.tolist(), df['evidence'].values.tolist()

        if self.is_train or self.tta:
            t1, t2 = self.get_aug(t1), self.get_aug(t2)

        pos = len(t1)
        ts = segment_shuffle(t1 + t2)
        t1, t2 = ts[:pos], ts[pos:]

        tokens = self.tokenize2(t1, t2)

        labels = [[label, 1 - label] for label in df['label']]

        tokens = {a: b.to(self.device) for a, b in tokens.items()}
        labels = torch.tensor(labels).type(torch.float).to(self.device)

        return tokens, labels

    @background(max_prefetch=32)
    def data_iter(self):
        if self.is_train:
            df = self.data['train_pandas']
            slices = self.data['train_slices']
        else:
            df = self.data['test_pandas']
            slices = self.data['test_slices']

        slices = sklearn.utils.shuffle(slices)[:self.cap]
        for b in slices:
            try:
                x, y = self.load_fn(df[b])
                yield x, y
            except:
                print(b)
                print(df[b])


class V6(DataMixin):

    def __init__(self):
        super().__init__()
        self.num_accumulation_steps = 1
        self.learning_rate = 1e-4
        self.epochs = 20

    def train(self, model, save='base_model.pth', test=False):

        print(datetime.datetime.now(), 'Train')

        print("Lr", self.learning_rate)

        optimizer = AdamW(model.parameters(), lr=self.learning_rate)

        loss_fn = CrossEntropyLoss()

        for epoch in range(self.epochs):
            print('======== Epoch {:} / {:} ========'.format(epoch + 1, self.epochs))

            scores = {'auc': [], 'acc': []}

            model.train()
            self.is_train = True

            grad_zero = True
            for i, data in enumerate(tqdm(self.data_iter(), total=len(self.data['train_slices']))):
                # model.zero_grad(set_to_none=True)

                logits, y = model.call(data)

                loss = loss_fn(logits, y)

                loss = loss / self.num_accumulation_steps
                loss.backward()

                pred = logits.cpu() > 0
                true = y.cpu()

                acc = accuracy_score(true, pred)

                scores['acc'].append(acc)

                if (i + 1) % self.num_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    grad_zero = True
                else:
                    grad_zero = False
            else:
                if grad_zero:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            print('Acc: %s' % np.mean(scores['acc']))

            if test:
                print('======== Test ========'.format(epoch + 1, self.epochs))
                self.test(model)

        self.is_train = False

    def test(self, model_or_path):
        print(datetime.datetime.now(), 'Test')

        if isinstance(model_or_path, str):
            model = torch.load(model_or_path)
        else:
            model = model_or_path

        model.eval()
        self.is_train = False

        scores = {'auc': [], 'acc': []}

        preds, trues = [], []

        with torch.no_grad():
            for data in tqdm(self.data_iter(), total=len(self.data['test_slices'])):
                logits, y = model.call(data)

                pred = logits > 0

                trues.extend(y.cpu().numpy().astype(int))
                preds.extend(pred.cpu().numpy().astype(int))

        trues, preds = np.array(trues), np.array(preds)

        acc = accuracy_score(trues, preds)

        print('Acc: %s' % np.mean(acc))

    def echo(self, model):
        print(datetime.datetime.now(), 'Echo')
        for data in self.data_iter():
            logits, y = model.call(data)
            print(logits)
            break



# self = V4()
# self.shuffle = False
# self.cap = 10000
# # self.device = "cpu"
# self.load_ds()
# self.epochs = 20
# # self.batch_size = 16
# # self.train(model)
