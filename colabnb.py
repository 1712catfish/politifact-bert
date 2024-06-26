# -*- coding: utf-8 -*-
"""politifact.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1kJvqnIpAo7u6JAASztGoDTLQ-1kCRIct
"""

!pip install nltk gensim --quiet
!pip install tqdm --quiet
!pip install transformers==4.37.2
!pip install gdown
!pip install torcheval
# !pip install torch_optimizer

!gdown 1--rt8hjab5YByCNRcv2s_8LaszZp6-5E
!gdown 1sAD2yLi5ujNfBGy3JCeI-uluDf_22Tdg
!git clone https://github.com/1712catfish/politifact-bert.git

# Commented out IPython magic to ensure Python compatibility.
# %cd politifact-bert
!git pull
# %cd /content

# Commented out IPython magic to ensure Python compatibility.
data = None
# %run -i politifact-bert/v3.py

class Project(Stable):
    def __init__(self):
        super().__init__()
        self.model_name = "bert-base-uncased"
        self.max_seq_len = 200
        self.batch_size = 16
        self.lr = 1e-4

self = Project()

if data is None:
    if self.data is None:
        self.load_ds()
    data = self.data
elif self.data is None:
    self.data = data

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

    def forward(self, input_ids, attention_mask, token_type_ids):
        # x = self.base(input_ids, attention_mask).hidden_states[-1]
        # # x = self.attn(x)
        # # x = x + self.dense1(self.norm2(x))
        # # x = self.pool(x)
        # x = x[:, 0]
        # # x = self.dense(first_token_tensor)
        # x = self.dense2(x)
        x = self.base(input_ids, attention_mask, token_type_ids).logits
        return x


model = NNModel().to(self.device)

self.echo(model)
self.train(model, 20, cap=self.batch_size)
self.echo(model)

data['train_inputs']['input_ids'][0]

data['train_inputs']['input_ids'][0]

