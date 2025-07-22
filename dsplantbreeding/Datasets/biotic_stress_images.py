from pathlib import Path

import tensorflow as tf


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

