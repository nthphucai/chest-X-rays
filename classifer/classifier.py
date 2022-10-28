import torch
import torch.nn as nn
import sys 
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import pandas as pd 
import numpy as np

from torchvision.models import densenet121

from reposcv.modules.commons import GlobalAverage
from reposcv.modules.attentions import SAModule
from reposcv.training.models.GCN import GCN
from classifer.chexnet import ChexNet

def classifier(gcn=True, pretrained_path=None, freeze_feature=False, n_class=14): 
    # features = densenet121(True).features
    # final_width = int(features[-1].num_features)
    # print('final_width', final_width)

    features = ChexNet(trained=True).backbone
    final_width = ChexNet(trained=True).head[-1].in_features
    # print('final_width', final_width)

    if gcn:
      embedding_path = "/Users/HPhuc/Practice/12. classification/vinbigdata/output/embeddings_14.npy"
      correlation_path = "/Users/HPhuc/Practice/12. classification/vinbigdata/output/correlations_14.npy"
      embeddings = np.load(embedding_path)
      corr_matrix = np.load(correlation_path)
      print("embeddings.shape", embeddings.shape)
      print("corr_matrix.shape", corr_matrix.shape)
      classifier = nn.Sequential(*[
                        GlobalAverage(),
                        GCN(300, final_width, embeddings, corr_matrix)
                      ])

    else:
      classifier = nn.Sequential(*[     
          GlobalAverage(keepdims=True),
          nn.Flatten(), 
          nn.Linear(final_width, n_class), 
          nn.Sigmoid()
         ])
    
    model = nn.Sequential()
    model.features = features
    model.dropout  = nn.Dropout(0.5)
    model.attention = SAModule(final_width)
    model.classifier = classifier
    
    if pretrained_path is not None:
        print(f"pretrained_path:", pretrained_path)
        cp = torch.load(pretrained_path, map_location="cpu")
        model.load_state_dict(cp['model_state_dict'])

    if freeze_feature:
      print("Freeze feature")
      for p in model.features.parameters():
          p.requires_grad = False

    return model

