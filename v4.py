from v3 import *
import torch.nn as nn

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

self.model_name = "bert-base-uncased"
self.max_seq_len = 200
self.batch_size = 16
self.lr = 1e-4

self.echo(model)
self.train(model, 20, cap=self.batch_size)
self.echo(model)
