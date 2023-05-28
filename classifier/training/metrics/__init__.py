from classifier.training.metrics._classify import (Accuracy, AUCScore, F1Score,
                                                   prec_recall_fscore_support)

__mapping__ = {
    "auc": AUCScore,
    "accuracy": Accuracy,
    "f1": F1Score,
    "prfs": prec_recall_fscore_support,
}
