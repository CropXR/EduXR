import seaborn as sns
from matplotlib import pyplot as plt
from dsplantbreeding.actions import get_true_labels_and_probs


from sklearn.metrics import accuracy_score, confusion_matrix


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