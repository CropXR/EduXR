from pathlib import Path

from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn as sns
from collections import Counter

def get_image_biotic_stress_dataset(healthy_dir: Path, infected_dirs: list[Path]):
    all_image_paths = []
    all_image_labels = []

    for file_path in healthy_dir.iterdir():
        if file_path.suffix == '.JPG':
            all_image_paths.append(str(file_path))
            all_image_labels.append(0)

    for infected_dir in infected_dirs:
        for file_path in infected_dir.iterdir():
            if file_path.suffix == '.JPG':
                all_image_paths.append(str(file_path))
                all_image_labels.append(1)

    # Convert lists to tensors
    all_image_paths = tf.constant(all_image_paths)
    all_image_labels = tf.constant(all_image_labels, dtype=tf.int32)

    # Create a dataset
    dataset = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))
    dataset = dataset.map(load_and_preprocess_image)
    dataset = dataset.shuffle(buffer_size=len(all_image_paths))
    return dataset

def load_and_preprocess_image(image_path, label):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [64, 64])
    img = img / 255.0  # Normalize pixel values
    return img, label

def show_confusion_matrix(my_model, dataset, batch_size=64):
    # Collect true labels and predictions from the validation dataset
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    true_labels = []
    predictions = []

    for images, labels in dataset:
        true_labels.extend(labels.numpy())
        predicted_probs = my_model.predict(images, verbose=0)
        predicted_classes = (predicted_probs > 0.5).astype(int) # Threshold for binary classification
        predictions.extend(predicted_classes.flatten())

    # Calculate the confusion matrix
    cm = confusion_matrix(true_labels, predictions)

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