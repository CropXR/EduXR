from collections import Counter
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