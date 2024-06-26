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


class DataMixin:
    def __init__(self):
        super().__init__()
        self.cap = -1
        self.batch_size = 16
        self.data = None
        self.max_seq_len = 200
        self.device = "cuda:0"
        self.tokenizer = None

    def encode_label(self, labels):
        labels = [[label, 1 - label] for label in labels]
        return labels

    def tokenize(self, t1, t2):
        if self.tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name, do_lower_case=True)

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

    def get_data(self, train=True):
        cap = self.cap
        batch_size = self.batch_size

        if train:
            x = {k: v[:cap] for k, v in self.data['train_inputs'].items()}
            y = self.data['train_labels'][:cap]
            slices = self.data['train_slices']
        else:
            x = {k: v for k, v in self.data['test_inputs'].items()}
            y = self.data['test_labels']
            slices = self.data['test_slices']

        return self.data_iter_v2(x, y, slices)

    def load_ds(self, train_df_path='politifact_train.csv',
                test_df_path='politifact_test.csv',
                device=True):
        print(datetime.datetime.now(), 'Load dataset')

        cap = self.cap

        print("Capped at", cap)

        df = pd.read_csv(train_df_path)[:cap]
        test_df = pd.read_csv(test_df_path)[:cap]

        self.data['df'] = pd.read_csv(train_df_path)[:cap]
        self.data['test_df'] = pd.read_csv(test_df_path)[:cap]

        # self.data['train_slices'] = self.get_slices(df)
        # self.data['test_slices'] = self.get_slices(test_df)
        #
        # self.data['train_inputs'] = self.tokenize(df['claim'], df['evidence'])
        # self.data['train_labels'] = self.encode_label(df['label'])
        #
        # self.data['test_inputs'] = self.tokenize(test_df['claim'], test_df['evidence'])
        # self.data['test_labels'] = self.encode_label(test_df['label'])
        #
        # data = self.data
        #
        # if device:
        #     try:
        #         for k, v in self.data.items():
        #             if 'inputs' in k:
        #                 self.data[k] = {a: b.to(self.device) for a, b in v.items()}
        #             elif 'labels' in k:
        #                 self.data[k] = torch.tensor(v).type(torch.float).to(self.device)
        #     except Exception as e:
        #         print(e)
        # return data

    def get_slices(self, df):
        x, a, b = df['claim'], df['evidence'], df['label']

        indices = [i for i, (x1, x2) in enumerate(zip(x[:-1], x[1:])) if x1 != x2]
        indices = [0] + indices + [len(x)]

        slices = [slice(x, y, 1) for x, y in zip(indices[:-1], indices[1:])]

        return slices

    def load_fn(self, df, train=True):
        tokens = self.tokenize(df['claim'], df['evidence'])
        labels = self.encode_label(df['label'])

        try:
            tokens = {a: b.to(self.device) for a, b in tokens.items()}
            labels = torch.tensor(labels).type(torch.float).to(self.device)
        except Exception as e:
            print(e)

        return tokens, labels

    def data_iter_v1(self, x, y, slices):
        batch = slice(0, 0)

        for s in tqdm(slices):

            if s.stop - batch.start < self.batch_size:
                batch = slice(batch.start, s.stop)
            else:
                b = batch
                batch = slice(batch.stop, s.stop)
                yield x['input_ids'][b], x['attention_mask'][b], x['token_type_ids'][b], y[b]

    def data_iter_v2(self, x, y, slices):
        slices = sklearn.utils.shuffle(slices)[:self.cap]
        for b in tqdm(slices):
            yield x['input_ids'][b], x['attention_mask'][b], x['token_type_ids'][b], y[b]


class V4(DataMixin):

    def __init__(self):
        super(V4, self).__init__()
        self.num_accumulation_steps = 1
        self.learning_rate = 1e-4
        self.epochs = 20

    def train(self, model, save='base_model.pth', test=False):

        print(datetime.datetime.now(), 'Train')

        print("Lr", self.learning_rate)

        optimizer = AdamW(model.parameters(), lr=self.learning_rate)

        batch_size = self.batch_size

        loss_fn = CrossEntropyLoss()

        for epoch in range(self.epochs):
            print('======== Epoch {:} / {:} ========'.format(epoch + 1, self.epochs))

            scores = {'auc': [], 'acc': []}

            model.train()

            grad_zero = True
            for i, data in enumerate(self.get_data(train=True)):
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

    def test(self, model_or_path):
        print(datetime.datetime.now(), 'Test')

        if isinstance(model_or_path, str):
            model = torch.load(model_or_path)
        else:
            model = model_or_path

        model.eval()

        scores = {'auc': [], 'acc': []}

        preds, trues = [], []

        with torch.no_grad():
            for data in self.get_data(train=False):
                logits, y = model.call(data)

                pred = logits > 0

                trues.extend(y.cpu().numpy().astype(int))
                preds.extend(pred.cpu().numpy().astype(int))

        trues, preds = np.array(trues), np.array(preds)

        acc = accuracy_score(trues, preds)

        print('Acc: %s' % np.mean(acc))

    def echo(self, model):
        print(datetime.datetime.now(), 'Echo')

        for data in self.get_data(train=True):
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