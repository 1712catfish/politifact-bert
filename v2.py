import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

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
from torch.optim import AdamW, Adam
# from torch_optimizer import Yogi
# from torcheval.metrics.functional import binary_auroc
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy, softmax
from torch.nn import CrossEntropyLoss
# from torcheval.metrics import *
# from torchmetrics.classification import MulticlassAUROC
from sklearn.utils import gen_batches
import sklearn
import datetime
import torch.nn as nn
import torch.nn.functional as F
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
import torch.nn as nn
import torch.nn.functional as F
# from torcheval.metrics import *
# from torchmetrics.classification import MulticlassAUROC
from sklearn.utils import gen_batches
from sklearn.metrics import roc_auc_score


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


class Project:

    def __init__(self):
        self.model_name = "bert-base-uncased"
        self.max_seq_len = 200
        self.batch_size = 64
        self.device = "cuda:0"

        self.tokenizer = BertTokenizer.from_pretrained(self.model_name, do_lower_case=True)
        nltk.download("stopwords")
        self.stop_words = set(nltk.corpus.stopwords.words("english"))

        self.train_inputs = None
        self.train_labels = None
        self.test_inputs = None
        self.test_labels = None

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
        # return {k: v.to(self.device) for k, v in tokens.items()}
        return tokens

    def encode_label(self, labels, categories=(True, False)):
        # labels = torch.Tensor([
        #     [1 - categories.index(label), categories.index(label)] for label in labels
        # ])
        labels = [[float(categories.index(label))] for label in labels]
        # return labels.to(self.device)
        return labels

    def ds_to_device(self, train_inputs, train_labels, test_inputs, test_labels):
        return (
            {k: v.to(self.device) for k, v in train_inputs.items()},
            torch.tensor(train_labels).to(self.device),
            {k: v.to(self.device) for k, v in test_inputs.items()},
            torch.tensor(test_labels).to(self.device)
        )

    def load_ds(self, train_df_path='politifact_train.csv',
                test_df_path='politifact_test.csv',
                cap=None, shuffle=False):
        print(datetime.datetime.now(), 'Load dataset')

        if cap is None:
            cap = -1

        df = pd.read_csv(train_df_path)[:cap]
        test_df = pd.read_csv(test_df_path)[:cap]

        SEP = " therefore therefore therefore therefore  "
        # SEP = self.tokenizer.sep_token

        df['input'] = df['evidence'] + SEP + df['claim']
        test_df['input'] = test_df['evidence'] + SEP + test_df['claim']

        inputs, labels = df['input'], df['label']

        if shuffle:
            inputs, labels = sklearn.utils.shuffle(df['input'], df['label'], random_state=123)
        else:
            inputs, labels = df['input'], df['label']

        train_inputs, train_labels = self.encode(inputs), self.encode_label(labels)

        test_inputs, test_labels = self.encode(test_df['input']), self.encode_label(test_df['label'])

        self.train_inputs, self.train_labels, self.test_inputs, self.test_labels = self.ds_to_device(train_inputs,
                                                                                                     train_labels,
                                                                                                     test_inputs,
                                                                                                     test_labels)

    def train(self, model, epochs=10, save='base_model.pth'):
        print(datetime.datetime.now(), 'Train')

        optimizer = AdamW(model.parameters(), lr=1e-4)

        batch_size = self.batch_size

        loss_fn = CrossEntropyLoss()

        x = self.train_inputs
        y = self.train_labels

        for epoch in range(epochs):
            print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
            loop = tqdm(list(gen_batches(len(y), batch_size)))
            scores = {'auc': [], 'acc': []}

            model.train()

            for batch in loop:
                model.zero_grad(set_to_none=True)

                logits = model(
                    x['input_ids'][batch],
                    x['attention_mask'][batch],
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

        with torch.no_grad():
            for test_batch in list(gen_batches(len(self.test_labels), batch_size)):
                outputs = model(
                    x['input_ids'][test_batch],
                    x['attention_mask'][test_batch],
                )

                pred = (outputs > 0).cpu().numpy().astype(int)[:, 0]
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
            x['attention_mask'][:self.batch_size])
        )

    def accuracy(self, logits, labels):
        return torch.mean((logits > 0) == labels, dtype=torch.float32)

    def AUC(self, logits, labels):
        scores = []
        for i in range(logits.shape[-1]):
            preds = torch.as_tensor(logits[:, i] > 0, dtype=torch.float32)
            scores += [binary_auroc(preds, labels[:, 0])]

            # auc += [preds == labels[:, 0]]

        return torch.mean(torch.as_tensor(scores, dtype=torch.float32))

    def baseline(self):
        class NNModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.base = BertForSequenceClassification.from_pretrained(
                    "bert-base-uncased",
                    num_labels=1,
                    output_hidden_states=True,
                    # output_attentions=True,
                    # hidden_dropout_prob=0.0,
                    # classifier_dropout=0.0,
                )
                self.attn = MultiheadAttention(768, 768, 2)
                self.dense1 = nn.Linear(768, 768)
                self.norm1 = nn.BatchNorm1d(200)
                self.norm2 = nn.BatchNorm1d(200)
                self.act = nn.ReLU()
                self.dense2 = nn.Linear(768, 1)
                self.pool = BertPooler(768)

            def forward(self, input_ids, attention_mask):
                x = self.base(input_ids, attention_mask).hidden_states[-1]
                x = x + self.attn(self.norm1(x))
                x = x + self.dense1(self.norm2(x))
                x = self.pool(x)
                x = self.dense2(x)
                return x

        return NNModel()

    # def train_test_split(self):
    # train_ds, test_ds = train_test_split(df, test_size=0.2, shuffle=True, random_state=123)

    # train_inputs, train_labels = self.encode(train_ds['claim']), self.prep_label(train_ds['label'])
    # test_inputs, test_labels = self.encode(test_ds['claim']), self.prep_label(test_ds['label'])


self = Project()

self.batch_size = 16
self.load_ds(cap=128)
model = self.baseline().to(self.device)
# self.echo(model)
# self.test(model)
self.train(model)