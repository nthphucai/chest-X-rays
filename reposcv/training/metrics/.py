import torch
import numpy as np

def dice(probs,
        truths,
        axes = (-1,-2), 
        threshold: float = 0.5,
        eps: float = 1e-9) -> np.ndarray:
    """
    Returns: Dice score for data batch.
      prob: model outputs after activation function.
      truth: truth values.
    """
    scores = []
    preds = (probs >= threshold).float()
    assert(preds.shape == truths.shape)
    intersection = 2.0 * (truths * preds).sum(axis=axes)
    union = truths.sum(axis=axes)+ preds.sum(axis=axes)
    if truths.sum(axis=axes)== 0 and preds.sum(axis=axes) == 0:
        scores.append(1.0)
    else:
      scores.append((intersection + eps) / union)
            
    return np.mean(scores)

class Dice_3D:
    def __init__(self, threshold: float = 0.5):
      self.threshold: float = threshold
      self.dice_class_liver: list = []
      self.dice_class_tumor: list = []
      self.dice_scores: list = [] 
    
    def init_list(self): 
      self.dice_class_liver = []
      self.dice_class_tumor = []
      self.dice_scores = [] 

    def update_class(self, logits: torch.Tensor, targets: torch.Tensor):
      """
      Takes: logits from output model and targets,
      calculates dice and iou scores, and stores them in lists.
      """
      probs = torch.sigmoid(logits)
      dice_liver, dice_tumor = self.dice_class(probs.detach().cpu().numpy(), targets.detach().cpu().numpy(), self.threshold)

      self.dice_class_liver.append(dice_liver)
      self.dice_class_tumor.append(dice_tumor)
            
    def get_dice_class(self) -> np.ndarray:
      """
      Returns: dice score per classes.
      """
      dice_class_liver = np.mean(self.dice_class_liver)
      dice_class_tumor = np.mean(self.dice_class_tumor)

      return dice_class_liver, dice_class_tumor

    def update_batch(self, logits: torch.Tensor, targets: torch.Tensor):  
      
      probs = torch.sigmoid(logits)
      dice = self.dice_batch(probs.detach().cpu().numpy(), targets.detach().cpu().numpy(), threshold = self.threshold)
      
      self.dice_scores.append(dice)
      
    def get_dice_batch(self) -> np.ndarray:
      """
      Returns: dice score per batch.
      """
      dice_score = np.mean(self.dice_scores)

      return dice_score

    @staticmethod
    def dice_batch(probs: np.ndarray,
                  truths: np.ndarray,
                  threshold: float = 0.5,
                  eps: float = 1e-9) -> np.ndarray:
        """
        Params:
          prob: model outputs after activation function = (batch, class, c, w, h)
          truths: model targets = (batch, class, c, w, h)
          Returns: Dice score for data batch.
        """
        scores = []
        num = probs.shape[0]
        preds = (probs >= threshold).astype(np.float32)
        assert(preds.shape == truths.shape)
        for i in range(num):
            pred = preds[i]
            truth = truths[i]
            intersection = 2.0 * (truth * pred).sum()
            union = truth.sum() + pred.sum()
            if truth.sum() == 0 and pred.sum() == 0:
                scores.append(1.0)
            else:
                scores.append((intersection + eps) / union)
                
        return np.mean(scores)

    @staticmethod
    def dice_class(probs: np.ndarray,
                  truths: np.ndarray,
                  threshold: float = 0.5,
                  eps: float = 1e-9,
                  classes: list = ['liver', 'tumor']) -> np.ndarray:

        """
        Returns: dict with dice scores for data batch and each class.    
            probs: model outputs after activation function = (batch, class, c, w, h)
            truths: model targets = (batch, class, c, w, h)
            classes: list with name classes.
        """
        scores = {key: list() for key in classes}
        num = probs.shape[0]
        num_classes = probs.shape[1]
        preds = (probs >= threshold).astype(np.float32)
        assert(preds.shape == truths.shape)

        for i in range(num):
            for class_ in range(num_classes):
                pred = preds[i][class_]
                truth = truths[i][class_]
                intersection = 2.0 * (truth * pred).sum()
                union = truth.sum() + pred.sum()
                if truth.sum() == 0 and pred.sum() == 0:
                    scores[classes[class_]].append(1.0)
                else:
                    scores[classes[class_]].append((intersection + eps) / union)
                    
        return np.mean(scores['liver']), np.mean(scores['tumor']) 
        