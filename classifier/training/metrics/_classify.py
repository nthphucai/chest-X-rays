import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support as pcfs
from sklearn.metrics import roc_auc_score


class Accuracy(nn.Module):
    def __init__(self, binary=False, classes=None):
        super().__init__()

        self.binary = binary
        if ".csv" in classes:
            self.classes = pd.read_csv(classes, index_col=0)
        elif ".npy" in classes:
            self.classes = np.load(classes)

    def forward(self, preds, trues):
        if self.binary:
            preds = preds >= 0.5
        else:
            preds = preds.argmax(dim=1)

        result = [(trues[:, c] == preds[:, c]).float() for c in range(trues.shape[1])]
        result = result[0]
        # result = (preds == trues).float().mean()

        if self.classes is not None:
            result = {f"{c}_ACC": result[i].mean() for i, c in enumerate(self.classes)}
            return result

        return result

    def extra_repr(self):
        return f"binary={self.binary}"


class F1Score(nn.Module):
    def __init__(self, binary=True, classes=None, average="binary", **kwargs):
        super().__init__()

        self.average = average
        self.binary = binary
        self.kwargs = kwargs

        if ".csv" in classes:
            self.classes = pd.read_csv(classes, index_col=0)
        elif ".npy" in classes:
            self.classes = np.load(classes)

    def forward(self, preds, trues):
        if self.binary:
            preds = (preds >= 0.5).float()
        else:
            trues = trues[:, 1:, ...]
            preds = preds.argmax(dim=1)
            preds = torch.stack(
                [(preds == i).float() for i in range(1, trues.shape[1])], dim=1
            )

        fb = torch.tensor(
            [
                f1_score(
                    trues[:, i],
                    preds[:, i],
                    average="binary",
                    zero_division=1,
                    **self.kwargs,
                )
                for i in range(trues.shape[1])
            ]
        )

        if self.classes is not None:
            fb = {f"{c}_F1": fb[i].mean() for i, c in enumerate(self.classes)}
            # fb[f"F1"] = fb.mean()
            return fb

        return {f"F1": fb.mean()}

    def extra_repr(self):
        return f"binary={self.binary}, " f"classes={self.classes}"


class prec_recall_fscore_support(nn.Module):
    def __init__(
        self,
        beta=1,
        binary=True,
        classes=None,
        metric="Fscore",
        average="binary",
        **kwargs,
    ):
        super().__init__()

        self.beta = beta
        self.average = average
        self.binary = binary
        self.metric = metric
        self.kwargs = kwargs

        if ".csv" in classes:
            self.classes = pd.read_csv(classes, index_col=0)
        elif ".npy" in classes:
            self.classes = np.load(classes)

    def forward(self, preds, trues):
        if self.binary:
            preds = (preds >= 0.5).float()
        else:
            trues = trues[:, 1:, ...]
            preds = preds.argmax(dim=1)
            preds = torch.stack(
                [(preds == i).float() for i in range(1, trues.shape[1])], dim=1
            )

        mtric = {"Precision": 0, "Recall": 1, "Fscore": 2, "Support": 3}
        results = [
            pcfs(
                trues[:, i],
                preds[:, i],
                average="binary",
                zero_division=1,
                **self.kwargs,
            )
            for i in range(trues.shape[1])
        ]
        if self.metric == "Fscore":
            result = torch.Tensor(
                [results[i][mtric["Fscore"]] for i in range(trues.shape[1])]
            )
        elif self.metric == "Precision":
            result = torch.Tensor(
                [results[i][mtric["Precision"]] for i in range(trues.shape[1])]
            )
        elif self.metric == "Recall":
            result = torch.Tensor(
                [results[i][mtric["Recall"]] for i in range(trues.shape[1])]
            )

        if self.classes is not None:
            result = {
                f"{c}_{self.metric}": result[i] for i, c in enumerate(self.classes)
            }
            return result

        return {f"{self.metric}": result.mean()}

    def extra_repr(self):
        return f"metric={self.metric}, binary={self.binary}, " f"classes={self.classes}"


class AUCScore(nn.Module):
    def __init__(self, binary=True, classes=None):
        super().__init__()

        self.binary = binary
        if ".csv" in classes:
            self.classes = pd.read_csv(classes, index_col=0)
        elif ".npy" in classes:
            self.classes = np.load(classes)

    def forward(self, preds, trues):
        if self.binary:
            preds = (preds >= 0.5).float()
        else:
            trues = trues[:, 1:, ...]
            preds = preds.argmax(dim=1)
            preds = torch.stack(
                [(preds == i).float() for i in range(1, trues.shape[1])], dim=1
            )

        auc_score = []

        for i, c in enumerate(self.classes):
            pred = preds[:, i]
            true = trues[:, i]
            if len(torch.unique(true)) == 2:
                auc_score.append(roc_auc_score(true, pred))
            elif len(torch.unique(true)) != 2:
                if torch.all(pred == true):
                    auc_score.append(1)
                else:
                    auc_score.append(1)

        results = {f"{c}_AUC": auc_score[i] for i, c in enumerate(self.classes)}

        return results

    def extra_repr(self):
        return f"binary={self.binary}, " f"classes={self.classes}"
