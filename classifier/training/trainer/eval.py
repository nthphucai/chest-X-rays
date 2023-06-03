import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm 

from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, roc_curve, auc

from classifier.models.classify import classifier


class ThoracicClassifierEvaluator:
    def __init__(self,
        model: nn.Module,
        data_loader: DataLoader,  
        binary: bool = True, 
        classes = None,
        gcn: bool=False
    ):
        
        super().__init__()

        self.classes = classes
        self.binary  = binary
        self.data_loader = data_loader

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)

    def compute_roc(self):
        trues, preds = self.evaluate()
        class_rocs = {f"{c}_ROC": roc_curve(trues[:, i], preds[:, i]) for i, c in enumerate(self.classes)}
        return class_rocs

    def evaluate(self):
        trues = []
        preds = []

        self.model.eval()
        with torch.no_grad():
          for img, true in tqdm(self.data_loader):
              img = img.to(self.device)
              true = true.to(self.device)
                  
              pred = self.model(img)
                  
              trues.append(true.detach().cpu()) 
              preds.append(pred.detach().cpu())

        trues = torch.cat(trues)
        preds = torch.cat(preds)

        avg_aucs = self.compute_scores(preds, trues)
        class_aucs = self.compute_scores(preds, trues)
        for key, value in zip(class_aucs.keys(), class_aucs.values()):
                print(key, ':' , f'{value:.3f}', end='\n') 

        return trues, preds
    
    def compute_scores(self, preds, trues):
          if self.binary:
              preds = (preds >= 0.5).float()
          else:
              trues = trues[:, 1:, ...]
              preds = preds.argmax(dim=1)
              preds = torch.stack([(preds == i).float() for i in range(1, trues.shape[1])], dim=1)

          auc_score = []
          for i in range (trues.shape[1]):
              pred = preds[:, i]
              true = trues[:, i]

              if len(torch.unique(true)) == 2:
                auc_score.append(roc_auc_score(true, pred))
              elif len(torch.unique(true)) != 2:
                if torch.all(pred == true): auc_score.append(1)
                else:  
                  auc_score.append(1)

          if self.classes is not None:
            results = {f"{c}_AUC": auc_score[i] for i, c in enumerate(self.classes)}
          else:
            results = np.mean(auc_score)

          return results   
