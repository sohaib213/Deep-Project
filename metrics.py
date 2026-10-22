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