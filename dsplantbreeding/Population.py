import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from dsplantbreeding.Models import GenomicSelectionModel

class PlantPopulation:
    def __init__(self, markers: pd.DataFrame, name="Population"):
        self._markers = markers  # DataFrame: rows = plants, cols = Markers
        self.name = name

    @property 
    def n_plants(self):
        return self._markers.shape[0]

    @property 
    def n_markers(self):
        return self._markers.shape[1]

    @property
    def _phenotypes(self):
        np.random.seed(12345)
        phenotype_dict = dict()
        # Simulate phenotype: Marker_12 has strong effect
        phenotype = 30 + self._markers["Marker_12"] * 50 + np.random.normal(0, 5, self.n_plants)
        phenotype_dict['Salt Resistance (% survival)'] = phenotype

        # Simulate phenotype: yield is particularly influenced by marker 1-10 and 20-23
        pos_yield_phenotype = self._markers[[f"Marker_{i}" for i in range(0,10)]].sum(axis=1) * 2 + np.random.normal(0, 1, self.n_plants)
        neg_yield_phenotype = self._markers[[f"Marker_{i}" for i in range(20,23)]].sum(axis=1) * -1 + np.random.normal(0, 1, self.n_plants)

        phenotype_dict['Yield (Kg/Ha)'] = pos_yield_phenotype + neg_yield_phenotype
        
        return pd.DataFrame.from_dict(phenotype_dict)

    def show_size(self):
        n_plants = len(self._markers)
        plant_word = "plant" if n_plants == 1 else "plants"
        print(f"{self.name} has {n_plants} {plant_word} and {self._markers.shape[1]} markers.")

    def show_marker_at_location(self, marker_index):
        print(self._markers.iloc[:, marker_index].values)

    def show_all_phenotypes(self):
        print(self._phenotypes)

    def show_phenotype(self, name: str):
        print(self._phenotypes[name])

    def head(self):
        print(pd.concat((self._markers, self._phenotypes), axis=1).head())

    def show_manhattan_plot(self, to_phenotype: str):
        phenotype_to_correlate_to = self._phenotypes[to_phenotype]
        p_values = self._markers.apply(lambda col: pearsonr(col, phenotype_to_correlate_to)[1], axis=0)
        neg_log_p = -np.log10(p_values)
        plt.figure(figsize=(10, 4))
        plt.bar(range(len(neg_log_p)), neg_log_p)
        plt.title("Manhattan Plot (-log10 p-values)")
        plt.xlabel("Marker index")
        plt.ylabel("-log10(p-value)")
        plt.show()

    def show_marker_to_phenotype_relation(self, marker_location, to_phenotype):
        marker_values = self._markers.iloc[:, marker_location]
        phenotype_values = self._phenotypes[to_phenotype]
        plt.scatter(marker_values, phenotype_values)
        plt.xlabel(f"Marker_{marker_location}")
        plt.ylabel(to_phenotype)
        plt.title(f"Relation between Marker_{marker_location} and {to_phenotype}")
        plt.show()

    def select_plants_with_marker_at_location(self, marker_index, desired_allele=1):
        mask = self._markers.iloc[:, marker_index] == desired_allele
        new_markers = self._markers[mask].reset_index(drop=True)
        return PlantPopulation(new_markers, name=f"Selected@Marker{marker_index}")

    def sample_plants(self, n_offspring):
        return self._markers.sample(n_offspring, replace=True).reset_index(drop=True)

    def fit_gs_model(self, target_phenotype) -> GenomicSelectionModel:
        return GenomicSelectionModel(self, target_phenotype)
    
    def select_plants_from_predicted_values(self, predicted_values, ntop=10):
        sorted_indices = np.argsort(predicted_values)[-ntop:]
        selected_markers = self._markers.iloc[sorted_indices].reset_index(drop=True)
        return PlantPopulation(selected_markers, name=f"SelectedTop{ntop}")


def get_resilient_population(n_plants=1, n_markers=50):
    np.random.seed(42)
    markers = pd.DataFrame(np.random.randint(0, 2, size=(n_plants, n_markers)),
                        columns=[f"Marker_{i}" for i in range(n_markers)])
    markers["Marker_12"] = 1  # Ensure Marker_12 is present for resilience

    return PlantPopulation(markers)


def get_agricultural_population(n_plants=1, n_markers=50):
    np.random.seed(420)
    markers = pd.DataFrame(np.random.randint(0, 2, size=(n_plants, n_markers)),
                        columns=[f"Marker_{i}" for i in range(n_markers)])
    markers["Marker_12"] = 0

    return PlantPopulation(markers)


def get_natural_population(n_plants=50, n_markers=50):
    np.random.seed(421)
    markers = pd.DataFrame(np.random.randint(0, 2, size=(n_plants, n_markers)),
                        columns=[f"Marker_{i}" for i in range(n_markers)])

    return PlantPopulation(markers)
