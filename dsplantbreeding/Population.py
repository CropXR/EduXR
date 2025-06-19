import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from dsplantbreeding.Models import GenomicSelectionModel

class PlantPopulation:
    def __init__(self, snps: pd.DataFrame, name="Population"):
        self._snps = snps  # DataFrame: rows = plants, cols = SNPs
        self.name = name

    @property 
    def n_plants(self):
        return self._snps.shape[0]

    @property 
    def n_snps(self):
        return self._snps.shape[1]

    @property
    def _phenotypes(self):
        np.random.seed(12345)
        phenotype_dict = dict()
        # Simulate phenotype: SNP_12 has strong effect
        phenotype = self._snps["SNP_12"] * 2 + np.random.normal(0, .1, self.n_plants)
        phenotype_dict['Salt Resistance'] = phenotype

        # Simulate phenotype: yield is particularly influenced by snp 1-10 and 20-25
        pos_yield_phenotype = self._snps[[f"SNP_{i}" for i in range(0,10)]].sum(axis=1) * 2 + np.random.normal(0, .5, self.n_plants)
        neg_yield_phenotype = self._snps[[f"SNP_{i}" for i in range(20,23)]].sum(axis=1) * -1 + np.random.normal(0, .5, self.n_plants)

        phenotype_dict['Yield'] = pos_yield_phenotype + neg_yield_phenotype
        
        return pd.DataFrame.from_dict(phenotype_dict)

    def show_size(self):
        print(f"{self.name} has {len(self._snps)} plants and {self._snps.shape[1]} SNPs.")

    def show_snps_at_location(self, snp_name):
        print(self._snps[snp_name].values)

    def show_phenotypes(self):
        print(self._phenotypes)

    def head(self):
        df = self._snps.copy()
        df['phenotype'] = self._phenotypes()
        print(df.head())

    def show_manhattan_plot(self, to_phenotype: str):
        phenotype_to_correlate_to = self._phenotypes[to_phenotype]
        # TODO check if this implementation is correct
        p_values = self._snps.apply(lambda col: pearsonr(col, phenotype_to_correlate_to)[1], axis=0)
        neg_log_p = -np.log10(p_values)
        plt.figure(figsize=(10, 4))
        plt.bar(range(len(neg_log_p)), neg_log_p)
        plt.title("Manhattan Plot (-log10 p-values)")
        plt.xlabel("SNP index")
        plt.ylabel("-log10(p-value)")
        plt.show()

    def select_plants_with_snp_at_location(self, snp_index, desired_allele=1):
        mask = self._snps.iloc[:, snp_index] == desired_allele
        new_snps = self._snps[mask].reset_index(drop=True)
        return PlantPopulation(new_snps, name=f"Selected@SNP{snp_index}")

    def show_genetic_composition(self):
        pass

    def sample_plants(self, n_offspring):
        return self._snps.sample(n_offspring, replace=True).reset_index(drop=True)

    def fit_gs_model(self, target_phenotype) -> GenomicSelectionModel:
        return GenomicSelectionModel(self, target_phenotype)
    
    def select_plants_from_predicted_values(self, predicted_values, ntop=10):
        sorted_indices = np.argsort(predicted_values)[-ntop:]
        selected_snps = self._snps.iloc[sorted_indices].reset_index(drop=True)
        return PlantPopulation(selected_snps, name=f"SelectedTop{ntop}")


def get_resilient_population(n_plants=1, n_snps=50):
    np.random.seed(42)
    snps = pd.DataFrame(np.random.randint(0, 2, size=(n_plants, n_snps)),
                        columns=[f"SNP_{i}" for i in range(n_snps)])
    snps["SNP_12"] = 1  # Ensure SNP_12 is present for resilience

    return PlantPopulation(snps)


def get_agricultural_population(n_plants=1, n_snps=50):
    np.random.seed(420)
    snps = pd.DataFrame(np.random.randint(0, 2, size=(n_plants, n_snps)),
                        columns=[f"SNP_{i}" for i in range(n_snps)])
    snps["SNP_12"] = 0

    return PlantPopulation(snps)

  

def get_natural_population(n_plants=50, n_snps=50):
    np.random.seed(421)
    snps = pd.DataFrame(np.random.randint(0, 2, size=(n_plants, n_snps)),
                        columns=[f"SNP_{i}" for i in range(n_snps)])

    return PlantPopulation(snps)
