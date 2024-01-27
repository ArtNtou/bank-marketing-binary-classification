import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve


# get confusion matrix and plot ROC graphs and AUC values
def confusion_plots(y_train, y_test, y_pred_train, y_pred_test):
    # ROC Curve and AUC Curve
    fpr_train, tpr_train, _ = roc_curve(y_train, y_pred_train)
    roc_auc_train = roc_auc_score(y_train, y_pred_train)

    fpr_test, tpr_test, _ = roc_curve(y_test, y_pred_test)
    roc_auc_test = roc_auc_score(y_test, y_pred_test)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_test)

    # Plot ROC Curves for tuned models
    plt.figure(figsize=(10, 6))
    plt.plot(fpr_train, tpr_train,
             label=f'Train (AUC = {roc_auc_train:.2f})', linestyle='--')
    plt.plot(fpr_test, tpr_test,
             label=f'Test (AUC = {roc_auc_test:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('(ROC) Curve')
    plt.legend()
    plt.show(block=True)

    plt.subplot(1, 2, 2)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    plt.show(block=True)

    return cm