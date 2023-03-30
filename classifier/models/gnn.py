import torch 
import torch.nn as nn 
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from classifier.models.modules.attentions import SAModule
from classifier.models.modules.commons import GlobalAverage

class GraphLinear(nn.Linear):

    def __init__(self, in_channels, out_channels, correlation_matrix, bias=True):
        """
        :param in_channels: size of input features
        :param out_channels: size of output features
        :param correlation_matrix: correlation matrix for information propagation
        :param bias: whether to use bias
        """
        super().__init__(in_channels, out_channels, bias)

        assert isinstance(correlation_matrix, nn.Parameter), "correlation must be nn.Parameter"

        self.correlation_matrix = correlation_matrix

    def forward(self, x):
        prop = torch.matmul(self.correlation_matrix, x)

        return super().forward(prop)

class GraphSequential(nn.Module):

    def __init__(self, node_embedding, *args):
        """
        :param node_embedding: embedding extracted from text, either numpy or torch tensor
        :param args: additional torch module for transformation
        """
        super().__init__()

        if not torch.is_tensor(node_embedding):
            node_embedding = torch.tensor(node_embedding, dtype=torch.float)

        self.embedding = nn.Parameter(node_embedding, requires_grad=False)
        self.sequential = nn.Sequential(*args)

    def forward(self):
        return self.sequential(self.embedding)

class GCN(nn.Module):

    def __init__(self, width, embeddings, corr_matrix):
        super().__init__()

        self.corr = nn.Parameter(torch.tensor(corr_matrix, dtype=torch.float), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(corr_matrix.shape[0]))
        
        bottleneck = width // 2
        self.embeddings = GraphSequential(embeddings, *[
            GraphLinear(300, bottleneck, self.corr, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            GraphLinear(bottleneck, width * 3 * 3, self.corr, bias=True),
        ])

    def forward(self, x):
        embeddings = self.embeddings()
        embeddings = embeddings.view(-1, x.shape[1], 3, 3)
        scores = F.conv2d(x, embeddings, bias=self.bias, padding=1)
#         scores = F.linear(x, embeddings, bias=self.bias)

        return scores
"""
hubconfig
"""
import inspect as insp
import sys

from importlib import import_module

class HubEntries:

    def __init__(self, absolute_path, module_name):
        sys.path.append(str(absolute_path))
        self.module = import_module(module_name)
        
    def load(self, entry_name, *args, **kwargs):
        """
        load a function from entry file

        :param entry_name: function name
        :param args: args to input into the function
        :param kwargs: kwargs to input into the function
        :return:
        """
        function = getattr(self.module, entry_name)

        assert insp.isfunction(function), f"{function} is not a function"
        return function(*args, **kwargs)

    def list(self):
        """
        list all available entries
        :return: list of entries
        """
        function_names = [name for name, _ in insp.getmembers(self.module, insp.isfunction)]

        return function_names

    def inspect(self, entry_name):
        """
        inspect args and kwargs of an entry
        :param entry_name: name of the entry
        :return: argspec object
        """
        function = getattr(self.module, entry_name)

        assert insp.isfunction(function), f"{function} is not a function"

        spec = insp.getfullargspec(function)
        return spec

def get_entries(path):
    """
    get enty point of a hub folder
    :param path: path to python module
    :return: HubEntries
    """
    path = Path(path)
    print(path)
    return HubEntries(path.parent, path.name.replace(".py", ""))

def classifier(config_id, embedding_path=None, correlation_path=None, feature_path=None, pretrained=False, freeze_feature=False, n_class=1): 
    entries = get_entries("/content/drive/MyDrive/Lung_Xray/classification/SDM/hubconfig")
    features = entries.load("chexnext", config_id=config_id, pretrained=pretrained)
    final_width = entries.load("config_dict", config_id=config_id)["stages_width"][-1][0]
    final_width = int(final_width)
    print('final_width:', final_width)
    
    if embedding_path is not None:
        embeddings = np.load(embedding_path)
        corr_matrix = np.load(correlation_path)
        print("embeddings.shape", embeddings.shape)
        print("corr_matrix.shape", corr_matrix.shape)
        classifier = nn.Sequential(*[
                GCN(final_width, embeddings, corr_matrix),
                GlobalAverage(),
                nn.Sigmoid()
        ])
    else:
        classifier = nn.Sequential(*[
            nn.Conv2d(final_width, n_class, 3, padding=1),
            GlobalAverage(),
            nn.Sigmoid() if n_class == 1 else nn.Identity(),
        ])
    
    model = nn.Sequential()
    model.features = features
    model.attention = SAModule(final_width)
    
    if feature_path is not None:
        w = torch.load(feature_path, map_location="cpu")
        #w = {k: w[k] for k in w if 'classifier' not in k}
        print(model.load_state_dict(w, strict=False))
        
    if freeze_feature:
        print("Freeze feature")
        for p in model.parameters():
            p.requires_grad = False
    
    model.classifier = classifier

    return model


# def classifier(config_id, embedding_path=None, correlation_path=None, feature_path=None, pretrained=True, freeze_feature=False, n_class=1): 
#     features = densenet121(True).features
#     final_width = features[-1].num_features
    
#     if embedding_path is not None:
#       embeddings = np.load(embedding_path)
#       corr_matrix = np.load(correlation_path)
#       print("embeddings.shape", embeddings.shape)
#       print("corr_matrix.shape", corr_matrix.shape)
#       classifier = nn.Sequential(*[
#               GCN(final_width, embeddings, corr_matrix),
#               GlobalAverage(),
#               nn.Sigmoid()
#       ])
#     else:
#         # classifier = nn.Sequential(*[
#         #     nn.Conv2d(final_width, n_class, 3, padding=1),
#         #     GlobalAverage(),
#         #     nn.Sigmoid() if n_class == 1 else nn.Identity()
#         #     ])

#       classifier = nn.Sequential(*[     
#           nn.AdaptiveAvgPool2d(1),
#           nn.Flatten(), 
#           nn.Linear(final_width, 256), 
#           # nn.Dropout(0.5),
#           nn.Linear(256, 14),
#           nn.Sigmoid()
#          ])
    
#     model = nn.Sequential()
#     model.features = features
#     model.attention = SAModule(final_width)
    
#     if feature_path is not None:
#         w = torch.load(feature_path, map_location="cpu")
#         #w = {k: w[k] for k in w if 'classifier' not in k}
#         print(model.load_state_dict(w, strict=False))
        
#     if freeze_feature:
#         print("Freeze feature")
#         for p in model.parameters():
#             p.requires_grad = False
    
#     model.classifier = classifier

#     return model

# #=======================================
# config_id = 2
# embedding_path = "/content/drive/MyDrive/Classification2D/Source/outputs/embeddings_14.npy"
# #print(np.load(embedding_path))
# correlation_path = "/content/drive/MyDrive/Classification2D/Source/outputs/correlations_14.npy"
# # print(np.load(correlation_path))
# model = classifier(config_id=None, embedding_path=embedding_path, correlation_path=correlation_path, feature_path=None, pretrained=False, freeze_feature=False, n_class=14)
# # model = classifier(config_id=None, embedding_path=None, correlation_path=correlation_path, feature_path=None, pretrained=False, freeze_feature=False, n_class=14)

# x = torch.rand(1,3,256,256)
# print(model(x).shape)
# print(model(x))
