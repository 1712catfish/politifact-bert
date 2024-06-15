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


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / np.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


def expand_mask(mask):
    assert mask.ndim >= 2, "Mask must be at least 2-dimensional with seq_length x seq_length"
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    while mask.ndim < 4:
        mask = mask.unsqueeze(0)
    return mask


class MultiheadAttention(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, _ = x.size()
        if mask is not None:
            mask = expand_mask(mask)
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o


class BertPooler(nn.Module):
    def __init__(self, hidden_size):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


def panik():
    raise Exception("panik")


class Library:
    @property
    def train_inputs(self):
        return self.data['train_inputs']

    @property
    def test_inputs(self):
        return self.data['test_inputs']

    @property
    def train_labels(self):
        return self.data['train_labels']

    @property
    def test_labels(self):
        return self.data['test_labels']

    def legacy(self):
        pass
        # def train_test_split(self):
        # train_ds, test_ds = train_test_split(df, test_size=0.2, shuffle=True, random_state=123)

        # train_inputs, train_labels = self.encode(train_ds['claim']), self.prep_label(train_ds['label'])
        # test_inputs, test_labels = self.encode(test_ds['claim']), self.prep_label(test_ds['label'])

        # SEP = " therefore therefore therefore therefore  "
        # SEP = self.tokenizer.sep_token

        # df['input'] = df['evidence'] + SEP + df['claim'] + SEP
        # test_df['input'] = test_df['evidence'] + SEP + test_df['claim'] + SEP

        # inputs, labels = df['input'], df['label']

        # if shuffle:
        #     inputs, labels = sklearn.utils.shuffle(df['input'], df['label'], random_state=123)
        # else:
        #     inputs, labels = df['input'], df['label']

    def seq_preprocess(self, sequence):
        processed_words = simple_preprocess(
            re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(#[A-Za-z0-9]+)|(\w+:\/\/\S+)", " ", sequence))
        processed_words = [word.lower() for word in processed_words if word.lower() not in self.stop_words]

        return ' '.join(processed_words)

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

    def shuffle(self):
        inputs = [[k, v] for k, v in self.data['train_inputs'].items()]
        labels = self.data['labels']

        *inputs[:][1], labels = sklearn.utils.shuffle(*inputs[:][1], labels, random_state=123)

        self.data['train_inputs'] = {k: v for k, v in inputs}
        self.data['labels'] = labels

    def accuracy(self, logits, labels):
        return torch.mean((logits > 0) == labels, dtype=torch.float32)

    def AUC(self, logits, labels):
        scores = []
        for i in range(logits.shape[-1]):
            preds = torch.as_tensor(logits[:, i] > 0, dtype=torch.float32)
            scores += [binary_auroc(preds, labels[:, 0])]

            # auc += [preds == labels[:, 0]]

        return torch.mean(torch.as_tensor(scores, dtype=torch.float32))

    def load_ds(self, train_df_path='politifact_train.csv',
                test_df_path='politifact_test.csv',
                device=True):
        print(datetime.datetime.now(), 'Load dataset')

        cap = self.cap
        shuffle = self.shuffle

        print("Capped at", cap)

        df = pd.read_csv(train_df_path)[:cap]

        if shuffle:
            print("Shuffle")
            evidence, claim, label = sklearn.utils.shuffle(df['evidence'], df['claim'], df['label'], random_state=123)
        else:
            evidence, claim, label = df['evidence'], df['claim'], df['label']

        self.data['train_inputs'] = self.tokenize(evidence, claim)
        self.data['train_labels'] = self.encode_label(label)

        test_df = pd.read_csv(test_df_path)[:cap]
        self.data['test_inputs'] = self.tokenize(test_df['evidence'], test_df['claim'])
        self.data['test_labels'] = self.encode_label(test_df['label'])

        data = self.data

        if device:
            for k, v in self.data.items():
                if 'inputs' in k:
                    self.data[k] = {a: b.to(self.device) for a, b in v.items()}
                elif 'labels' in k:
                    self.data[k] = torch.tensor(v).type(torch.float).to(self.device)
                else:
                    panik()

        return data

    def train(self, model, epochs=20, cap=-1, save='base_model.pth'):
        print(datetime.datetime.now(), 'Train')

        optimizer = AdamW(model.parameters(), lr=self.lr)

        batch_size = self.batch_size

        loss_fn = CrossEntropyLoss()

        x = {k: v[:cap] for k, v in self.train_inputs.items()}
        y = self.train_labels[:cap]

        size = min(len(y), len(x['input_ids']))

        for epoch in range(epochs):
            print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
            loop = tqdm(list(gen_batches(size, batch_size)))
            scores = {'auc': [], 'acc': []}

            model.train()

            for batch in loop:
                model.zero_grad(set_to_none=True)

                logits = model(
                    x['input_ids'][batch],
                    x['attention_mask'][batch],
                    x['token_type_ids'][batch]
                )

                loss = loss_fn(logits, y[batch])
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                # with torch.no_grad():
                pred = logits.cpu() > 0
                true = y[batch].cpu()

                acc = accuracy_score(true, pred)
                # auc = roc_auc_score(true, pred, labels=(True, False))

                scores['acc'].append(acc)
                # scores['auc'].append(auc)
                loop.set_description(f'Acc: {acc:.2f}')

            print('Acc: %s' % np.mean(scores['acc']))
            # print('AUC: %s' % np.mean(scores['auc']))

            # torch.save(model, save)

            print('======== Test ========'.format(epoch + 1, epochs))
            self.test(model)

    def test(self, model_or_path):
        print(datetime.datetime.now(), 'Test')

        if isinstance(model_or_path, str):
            model = torch.load(model_or_path)
        else:
            model = model_or_path

        batch_size = self.batch_size
        x, y = self.test_inputs, self.test_labels.cpu()
        preds = []

        size = min(len(y), len(x['input_ids']))

        with torch.no_grad():
            for test_batch in tqdm(list(gen_batches(size, batch_size))):
                logits = model(
                    x['input_ids'][test_batch],
                    x['attention_mask'][test_batch],
                    x['token_type_ids'][test_batch]
                )

                pred = (logits > 0).cpu().numpy().astype(int)
                preds.extend(pred)

        acc = accuracy_score(y, preds)
        auc = roc_auc_score(y, preds, labels=(True, False))

        print('Acc: %s' % np.mean(acc))
        print('AUC: %s' % np.mean(auc))


    def echo(self, model):
        print(datetime.datetime.now(), 'Echo')

        x = self.train_inputs
        print(model(
            x['input_ids'][:self.batch_size],
            x['attention_mask'][:self.batch_size],
            x['token_type_ids'][:self.batch_size]
        ))

    def fit_one_batch(self, model, epochs=10):
        self.train(model, epochs, cap=self.batch_size)

    def __init__(self):
        self.device = "cuda:0"
        self.tokenizer = None
        nltk.download("stopwords")
        self.stop_words = set(nltk.corpus.stopwords.words("english"))

        self.model_name = "bert-base-uncased"
        self.max_seq_len = 200
        self.batch_size = 64

        self.data = {}

        self.cap = -1
        self.shuffle = True


# %run -i politifact-bert/v3.py

class Project(Library):
    def __init__(self):
        super().__init__()
        self.model_name = "bert-base-uncased"
        self.max_seq_len = 200
        self.batch_size = 16
        self.lr = 1e-4

# self = Project()
# print('Created self')
#
# reload = True
# if reload:
#     self.load_ds()
#     reload = False
# print('Created data')
