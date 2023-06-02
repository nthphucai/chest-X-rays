import numpy as np
import torch
import torch.nn as nn
from torchvision.models import densenet121

from .chexnet import ChexNet
# from .gcn import GCN
from .gnn import GCN
from .modules.attentions import SAModule
from .modules.commons import GlobalAverage


def classifier(
    backbone: str = None,
    gcn=True,
    pretrained_path=None,
    freeze_feature=False,
    n_class=14,
):
    if "densenet121" in backbone:
        features = densenet121(True).features
        final_width = int(features[-1].num_features)

    elif "chexnet" in backbone:
        features = ChexNet(trained=True).backbone
        final_width = ChexNet(trained=True).head[-1].in_features

    features = nn.Sequential(nn.Conv2d(1, 3, 3), features)
    print("final_width", final_width)

    if gcn:
        embedding_path = "data/vinbigdata/vin_classes_14.npy"
        correlation_path = "data/vinbigdata/vin_correlations_14.npy"
        embeddings = np.load(embedding_path)
        corr_matrix = np.load(correlation_path)
        print("embeddings.shape", embeddings.shape)
        print("corr_matrix.shape", corr_matrix.shape)
        classifier = nn.Sequential(
            *[GlobalAverage(), GCN(300, final_width, embeddings, corr_mclearatrix)]
        )

    else:
        classifier = nn.Sequential(
            *[
                GlobalAverage(keepdims=True),
                nn.Flatten(),
                nn.Linear(final_width, n_class),
                nn.Sigmoid(),
            ]
        )

    model = nn.Sequential()
    model.features = features
    model.dropout = nn.Dropout(0.5)
    model.attention = SAModule(final_width)
    model.classifier = classifier

    if pretrained_path is not None:
        print(f"pretrained_path:", pretrained_path)
        cp = torch.load(pretrained_path, map_location="cpu")
        model.load_state_dict(cp["model_state_dict"])

    if freeze_feature:
        print("Freeze feature")
        for p in model.features.parameters():
            p.requires_grad = False
    return model


def run_test_model():
    x = torch.rand(2, 1, 32, 32)
    model = classifier(gcn=False)
    out = model(x)
    print(out.shape)
