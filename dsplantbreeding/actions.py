from dsplantbreeding.Population import PlantPopulation
import pandas as pd
import numpy as np

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