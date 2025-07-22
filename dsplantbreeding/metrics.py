import seaborn as sns
from matplotlib import pyplot as plt
from dsplantbreeding.actions import get_true_labels_and_probs


from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_curve


def show_accuracy(my_model, dataset):
    true_labels, predicted_probs = get_true_labels_and_probs(my_model, dataset)
    predicted_classes = (predicted_probs > 0.5).astype(int) # Threshold for binary classification

    print(f'Accuracy: {accuracy_score(true_labels, predicted_classes):.2f}')


def show_confusion_matrix(my_model, dataset):
    true_labels, predicted_probs = get_true_labels_and_probs(my_model, dataset)

    predicted_classes = (predicted_probs > 0.5).astype(int) # Threshold for binary classification

    # Calculate the confusion matrix
    cm = confusion_matrix(true_labels, predicted_classes)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Healthy (0)', 'Infected (1)'],
                yticklabels=['Healthy (0)', 'Infected (1)']
                )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()


def show_auroc(model, dataset):
    # Prepare data for AUROC curve
    true_labels, predicted_probs = get_true_labels_and_probs(model, dataset)

    # Calculate the ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(true_labels, predicted_probs)
    roc_auc = auc(fpr, tpr)

    # Plot the AUROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (AUROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

    print(f"AUROC on validation set: {roc_auc:.4f}")