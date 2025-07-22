from collections import Counter

from matplotlib import pyplot as plt
from sklearn.metrics import auc, roc_curve, confusion_matrix
import seaborn as sns
from dsplantbreeding.Population import PlantPopulation
import pandas as pd
import numpy as np
import tensorflow as tf

def perform_cross_between(p1: PlantPopulation, p2: PlantPopulation, n_offspring=100, name=None):
    np.random.seed(102)
    # Assumes same number of markers, random recombination
    plants_1 = p1.sample_plants(n_offspring)
    plants_2 = p2.sample_plants(n_offspring)

    offspring_dict = dict()
    for col in plants_1.columns:
        alleles = np.where(np.random.rand(n_offspring) < 0.5,
                           plants_1[col],
                           plants_2[col])
        offspring_dict[col] = alleles

    offspring_df = pd.DataFrame(offspring_dict)

    return PlantPopulation(offspring_df, name=name)

def make_stress_pulse(start, end):
    return lambda t: 1.0 if start < t < end else 0.0


def decrease_brightness_on_label(target_label):
    def fn(img, label):
        img = tf.cond(
            tf.equal(label, target_label),
            lambda: tf.image.adjust_brightness(img, delta=-0.2),
            lambda: img
        )
        return img, label
    return fn

def augment_image(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.3)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return image, label

def count_labels_in_dataset(dataset):
    label_counter = Counter()
    for _, label in dataset:
        label_counter[f'Label={int(label.numpy())}'] += 1

    print(label_counter)


def show_confusion_matrix_with_examples(my_model, dataset, batch_size=64, max_per_category=4):
    true_labels = []
    predictions = []
    raw_probs = []
    image_store = []

    # Collect predictions and true labels
    for images, labels in dataset.batch(batch_size):
        probs = my_model.predict(images, verbose=0).flatten()
        preds = (probs > 0.5).astype(int)

        true_labels.extend(labels.numpy())
        predictions.extend(preds)
        raw_probs.extend(probs)
        image_store.extend(images.numpy())

    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Healthy (0)', 'Infected (1)'],
                yticklabels=['Healthy (0)', 'Infected (1)'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    # Classify each prediction into TP, FP, FN, TN
    buckets = {'TP': [], 'FP': [], 'FN': [], 'TN': []}
    for true, pred, prob, img in zip(true_labels, predictions, raw_probs, image_store):
        if true == 1 and pred == 1:
            buckets['TP'].append((img, true, pred, prob))
        elif true == 0 and pred == 0:
            buckets['TN'].append((img, true, pred, prob))
        elif true == 0 and pred == 1:
            buckets['FP'].append((img, true, pred, prob))
        elif true == 1 and pred == 0:
            buckets['FN'].append((img, true, pred, prob))

    # Plot the images grouped by confusion matrix category
    plt.figure(figsize=(max_per_category * 3, 12))
    for row_idx, category in enumerate(['TP', 'FP', 'FN', 'TN']):
        samples = buckets[category][:max_per_category]
        for col_idx, (img, true, pred, prob) in enumerate(samples):
            plt_idx = row_idx * max_per_category + col_idx + 1
            plt.subplot(4, max_per_category, plt_idx)
            plt.imshow(img)
            plt.title(f"{category}\nTrue:{true} Predicted:{pred}\nProb:{prob:.2f}")
            plt.axis("off")

    plt.suptitle("Example Predictions by Confusion Matrix Category", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def get_true_labels_and_probs(model, dataset, batch_size=32):
    true_labels = []
    predicted_probs = []

    for images, labels in dataset.batch(batch_size):
        probs = model.predict(images, verbose=0).flatten()
        true_labels.extend(labels.numpy())
        predicted_probs.extend(probs)

    return true_labels, predicted_probs


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


def preview_images(my_dataset):
    label_counter = Counter()
    plt.figure(figsize=(10, 10))
    index = 1
    for (img, label) in my_dataset:
        label = label.numpy()
        # Balance between both classes
        if label_counter[label] < 8:
            label_counter[label] += 1
        else:
            continue
        plt.subplot(4, 4, index)
        index += 1
        plt.imshow(img)
        plt.title(f"Label: {label}")
        plt.axis("off")
    plt.show()