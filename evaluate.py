import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score, roc_auc_score

    
def evaluate(model, test_data, test_labels, batch_size):
    
    # Compute predictions and round to obtain binary predictions
    yhat = np.round(model.predict(test_data, batch_size=batch_size))

    # Compute accuracy, confusion matrix, and metrics from confusion matrix
    acc = accuracy_score(test_labels, yhat) * 100
    tn, fp, fn, tp = confusion_matrix(test_labels, yhat).ravel()
    precision, recall, f1_score, _ = precision_recall_fscore_support(test_labels, yhat, average='binary')
    auroc = roc_auc_score(test_labels, yhat)

    # Print the results
    print("\nModel evaluation\n") 
    print(f"Confusion Matrix:\n{confusion_matrix(test_labels, yhat)}\n")
    print(f"Accuracy: {acc}%")
    print(f"Precision: {precision * 100}%")
    print(f"Recall: {recall * 100}%")
    print(f"F1 Score: {f1_score * 100}")
    print(f"AUROC: {auroc * 100}")


    