import numpy as np

def evaluate_metrics(y_true, y_pred):
    TP = TN = FP = FN = 0
    for true, pred in zip(y_true, y_pred):
        if true == 1 and pred == 1: TP += 1
        elif true == -1 and pred == -1: TN += 1
        elif true == -1 and pred == 1: FP += 1
        elif true == 1 and pred == -1: FN += 1
        
    accuracy = (TP + TN) / len(y_true) if len(y_true) > 0 else 0
    confusion_matrix = [[TP, FP], [FN, TN]]
    return accuracy, confusion_matrix

def evaluate_metrics_multi(y_true, y_pred, num_classes=3):
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    correct = 0

    for true, pred in zip(y_true, y_pred):
        t_idx = np.argmax(true)
        p_idx = np.argmax(pred)

        confusion_matrix[t_idx][p_idx] += 1
        if t_idx == p_idx:
            correct += 1

    accuracy = correct / len(y_true) if len(y_true) > 0 else 0
    return accuracy, confusion_matrix
