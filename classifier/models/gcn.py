import math 
import torch
import torch.nn as nn 
import numpy as np
from torchvision.models import densenet121
import torch.nn.functional as func

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GraphLinear(nn.Module):
    def __init__(self, in_channels, out_channels, adj_mtrix, bias=False):
        super().__init__()
        self.weight    = nn.Parameter(torch.Tensor(in_channels, out_channels), requires_grad=True)
        self.adj_mtrix = nn.Parameter(torch.tensor(adj_mtrix, dtype=torch.float), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels), requires_grad=True)
        else:
            self.register_parameter('bias', None)
            
        self.reset_params()

    def reset_params(self):
        stdv = 1./math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, embedding_nodes):
        support = torch.matmul(embedding_nodes.float(), self.weight.float())
        output = func.linear(self.adj_mtrix, support.T, self.bias) 
        return output

class GraphSequential(nn.Module):
    def __init__(self, embedding_nodes, *args):
        super().__init__()
        if not torch.is_tensor(embedding_nodes):
            embedding_nodes = torch.tensor(embedding_nodes, dtype=float)

        self.embedding_nodes = nn.Parameter(embedding_nodes, requires_grad=False)
        self.sequential = nn.Sequential(*args)

    def forward(self):
        return self.sequential(self.embedding_nodes)   # self.embedding_notes là input của self.sequential()

class GCN(nn.Module):
    def __init__(self, in_channels, out_channels, embedding_nodes, adj_mtrix, bias=True):
        super().__init__()
        
        bottleneck  = int(np.round(out_channels // 2))
        self.embedding = GraphSequential(embedding_nodes, *[
            GraphLinear(in_channels, bottleneck, adj_mtrix, bias=True),
            nn.LeakyReLU(0.2),
            GraphLinear(bottleneck, out_channels, adj_mtrix, bias=True),
        ])
        self.sigm = nn.Sigmoid()

    def forward(self, features):
        embedding = self.embedding().to(device)
        features = features.to(device)
        """ bias: True -> squeeze()
        """
        embedding = embedding.squeeze() 
        embedding = torch.transpose(embedding, 0,1)
        scores = torch.matmul(features, embedding)
        scores = self.sigm(scores)
        return scores 
