import numpy as np
import torch


def dice(
    probs: np.ndarray,
    truths: np.ndarray,
    axes=(-1, -2, -3, -4),
    threshold: float = 0.5,
    eps: float = 1e-9,
) -> np.ndarray:
    """
    Returns: Dice score for data batch.
      prob: model outputs after activation function.
      truth: truth values.
    """
    scores = []
    preds = (probs >= threshold).astype(np.float32)
    assert preds.shape == truths.shape
    intersection = 2.0 * (truths * preds).sum(axis=axes)
    union = truths.sum(axis=axes) + preds.sum(axis=axes)
    if truths.sum(axis=axes) == 0 and preds.sum(axis=axes) == 0:
        scores.append(1.0)
    else:
        scores.append((intersection + eps) / union)

    return np.mean(scores)


class Dice_3D:
    def __init__(self, threshold: float = 0.5):
        self.threshold: float = threshold
        self.dsc_class_organ: list = []
        self.dsc_class_tumor: list = []
        self.dsc_scores: list = []

    def init_list(self):
        self.dsc_class_organ = []
        self.dsc_class_tumor = []
        self.dsc_scores = []

    def update_class(self, probs: torch.Tensor, targets: torch.Tensor):
        """
        Takes: probs from output model and targets,
        calculates dice and iou scores, and stores them in lists.
        """
        dsc_organ, dsc_tumor = self.dice_class(
            probs.detach().cpu().numpy(), targets.detach().cpu().numpy(), self.threshold
        )

        self.dsc_class_organ.append(dsc_organ)
        self.dsc_class_tumor.append(dsc_tumor)

    def get_dice_class(self) -> np.ndarray:
        """
        Returns: dice score per classes.
        """
        dsc_class_organ = np.mean(self.dsc_class_organ)
        dsc_class_tumor = np.mean(self.dsc_class_tumor)

        return dsc_class_organ, dsc_class_tumor

    def update_batch(self, probs: torch.Tensor, labels: torch.Tensor):
        dice = self.dice_batch(
            probs.detach().cpu().numpy(),
            labels.detach().cpu().numpy(),
            threshold=self.threshold,
        )

        self.dsc_scores.append(dice)

    def get_dice_batch(self) -> np.ndarray:
        """
        Returns: dice score per batch.
        """
        dsc_scores = np.mean(self.dsc_scores)

        return dsc_scores

    @staticmethod
    def dice_batch(
        probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5, eps: float = 1e-9
    ) -> np.ndarray:
        """
        Params:
          prob: model outputs after activation function = (batch, class, c, w, h)
          truths: model targets = (batch, class, c, w, h)
          Returns: Dice score for data batch.
        """
        scores = []
        num = probs.shape[0]
        preds = (probs >= threshold).astype(np.float32)
        assert preds.shape == labels.shape
        for i in range(num):
            pred = preds[i]
            truth = labels[i]
            intersection = 2.0 * (truth * pred).sum()
            union = truth.sum() + pred.sum()
            if truth.sum() == 0 and pred.sum() == 0:
                scores.append(1.0)
            else:
                scores.append((intersection + eps) / union)

        return np.mean(scores)

    @staticmethod
    def dice_class(
        probs: np.ndarray,
        labels: np.ndarray,
        threshold: float = 0.5,
        eps: float = 1e-9,
        classes: list = ["organ", "tumor"],
    ) -> np.ndarray:
        """
        Returns: dict with dice scores for data batch and each class.
            probs: model outputs after activation function = (batch, class, c, w, h)
            labels: model targets = (batch, class, c, w, h)
            classes: list with name classes.
        """
        scores = {key: list() for key in classes}
        num = probs.shape[0]
        num_classes = probs.shape[1]
        preds = (probs >= threshold).astype(np.float32)
        assert preds.shape == labels.shape

        for i in range(num):
            for class_ in range(num_classes):
                pred = preds[i][class_]
                label = labels[i][class_]
                intersection = 2.0 * (label * pred).sum()
                union = label.sum() + pred.sum()
                if label.sum() == 0 and pred.sum() == 0:
                    scores[classes[class_]].append(1.0)
                else:
                    scores[classes[class_]].append((intersection + eps) / union)

        return np.mean(scores["organ"]), np.mean(scores["tumor"])
