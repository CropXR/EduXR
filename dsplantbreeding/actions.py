from dsplantbreeding.Population import PlantPopulation
import pandas as pd
import numpy as np

def perform_cross_between(p1: PlantPopulation, p2: PlantPopulation, n_offspring=100, name=None):
    # Assumes same number of SNPs, random recombination
    plants_1 = p1.sample_plants(n_offspring)
    plants_2 = p2.sample_plants(n_offspring)

    offspring_snps = pd.DataFrame()
    for col in plants_1.columns:
        alleles = np.where(np.random.rand(n_offspring) < 0.5,
                           plants_1[col],
                           plants_2[col])
        offspring_snps[col] = alleles

    return PlantPopulation(offspring_snps, name=name)