from classifier.training.metrics.fscore import (Accuracy, AUCScore, F1Score,
                                                prec_recall_fscore_support)

metric_maps = {
    "auc": AUCScore,
    "accuracy": Accuracy,
    "f1": F1Score,
    "prfs": prec_recall_fscore_support,
}
