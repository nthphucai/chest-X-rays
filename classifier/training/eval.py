from typing import Union, List

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm


class ThoracicClassifierEvaluator:
    def __init__(
        self,
        data_loader: DataLoader,
        binary: bool,
        classes: Union[np.array, List]
    ):
        self.binary = binary
        self.classes = classes
        self.data_loader = data_loader

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def evaluate(self, model: nn.Module):
        trues = []
        preds = []

        model.eval()
        model = model.to(self.device)

        with torch.no_grad():
            for img, true in tqdm(self.data_loader):
                img = img.to(self.device)
                true = true.to(self.device)

                pred = model(img)

                trues.append(true.detach().cpu())
                preds.append(pred.detach().cpu())

        trues = torch.cat(trues)
        preds = torch.cat(preds)

        class_aucs = self.compute_metrics(preds, trues)
        for key, value in zip(class_aucs.keys(), class_aucs.values()):
            print(key, ':', f'{value:.3f}', end='\n')

        return trues, preds

    def compute_roc(self, model: nn.Module):
        trues, preds = self.evaluate(model=model)
        class_rocs = {f"{c}_ROC": roc_curve(trues[:, i], preds[:, i]) for i, c in enumerate(self.classes)}
        return class_rocs

    def compute_metrics(self, preds, trues):
        if self.binary:
            preds = (preds >= 0.5).float()
        else:
            trues = trues[:, 1:, ...]
            preds = preds.argmax(dim=1)
            preds = torch.stack([(preds == i).float() for i in range(1, trues.shape[1])], dim=1)

        auc_score = []
        for i in range(trues.shape[1]):
            pred = preds[:, i]
            true = trues[:, i]

            if len(torch.unique(true)) == 2:
                auc_score.append(roc_auc_score(true, pred))
            elif len(torch.unique(true)) != 2:
                auc_score.append(1)

        if self.classes is not None:
            results = {f"{c}_AUC": auc_score[i] for i, c in enumerate(self.classes)}
        else:
            results = np.mean(auc_score)

        return results
