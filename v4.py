from v3 import *
import torch.nn as nn

class NNModel(nn.Module):

    def set_attn(self, layer, input, output):
        output = torch.mean(output, 1)
        self.attn = output

    def set_value(self, layer, input, output):
        self.value = output

    def __init__(self):

        super().__init__()

        self.backbone = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=1,
        )

        self.last_attn_layer = (
            self.backbone
            ._modules['bert']
            ._modules['encoder']
            ._modules['layer']
            ._modules['11']
            ._modules['attention']
            ._modules['self']
        )

        self.hdle1 = self.last_attn_layer._modules['dropout'].register_forward_hook(self.set_attn)
        self.hdle2 = self.last_attn_layer._modules['value'].register_forward_hook(self.set_value)

        self.gat_conv = GATv2Conv(-1, 100, heads=1, edge_dim=1)
        self.fc = nn.Linear(100, 2)

    def forward(self, *args):
        self.backbone(*args)

        nodes_batch = self.value
        weights_batch = self.attn

        outputs = []

        for nodes, weights in zip(nodes_batch, weights_batch):

            topk = torch.topk(weights, 100).values
            threshold = torch.min(topk)

            edges = weights > threshold
            edges = edges.nonzero().t()

            row, col = edges
            weights = weights[row, col]

            nodes = self.gat_conv(nodes, edges, weights)
            nodes = nn.ReLU()(nodes)

            nodes = nodes[0]
            nodes = nodes.unsqueeze(0)

            outputs.append(nodes)

        outputs = torch.cat(outputs)

        outputs = self.fc(outputs)

        return outputs

model = NNModel().to(self.device)

self.echo(model)

class NNModel(nn.Module):

    def set_attn(self, layer, input, output):
        output = torch.mean(output, 1)
        self.attn = output

    def set_value(self, layer, input, output):
        self.value = output

    def __init__(self):

        super().__init__()

        self.bb = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=1,
        )

        self.last_attn_layer = (
            self.bb
            ._modules['bert']
            ._modules['encoder']
            ._modules['layer']
            ._modules['11']
            ._modules['attention']
            ._modules['self']
        )

        self.last_attn_layer._modules['dropout'].register_forward_hook(self.set_attn)
        self.last_attn_layer._modules['value'].register_forward_hook(self.set_value)

        self.gat_conv = GATv2Conv(-1, 100, heads=1, edge_dim=1)
        self.fc = nn.Linear(100, 2)

    def forward(self, *input):
        self.bb(*input)

        nodes_batch = self.value
        weights_batch = self.attn

        outputs = []

        input_ids = input[0]

        # first_sep_token_pos = (input_ids == 102).nonzero()[0, 1]

        # bridge = torch.zero(s)

        # nodes_batch[:, s]

        # length = self.max_seq_len

        # for nodes in nodes_batch:
        #     graph = torch.stack(graph, nodes)

        #     edges = edges + [0:7]


        for nodes, weights in zip(nodes_batch, weights_batch):

            topk = torch.topk(weights, 100).values
            threshold = torch.min(topk)

            edges = weights > threshold
            edges = edges.nonzero().t()

            row, col = edges
            weights = weights[row, col]

            print(edges)

            nodes = self.gat_conv(nodes, edges, weights)
            nodes = nn.ReLU()(nodes)

            nodes = nodes[0]
            nodes = nodes.unsqueeze(0)

            outputs.append(nodes)

        outputs = torch.cat(outputs)

        outputs = self.fc(outputs)

        return outputs

model = NNModel().to(self.device)

self.echo(model)