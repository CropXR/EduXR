from sklearn.linear_model import LinearRegression, Ridge
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import sympy as sp
from scipy.integrate import odeint
import tensorflow as tf


class GenomicSelectionModel:
    def __init__(self, population, target_phenotype: str):
        # Population should be a PlantPopulation instance
        self.population = population
        self.target_phenotype = self.population._phenotypes[target_phenotype]
        self.model = None
        self.fit()
    
    def fit(self):
        X = self.population._markers.values
        y = self.target_phenotype
        self.model = Ridge(alpha=1.0)  
        self.model.fit(X, y)

    def show_genomic_selection_snp_weights(self):
        plt.figure(figsize=(10, 4))
        plt.bar(range(len(self.model.coef_)), self.model.coef_)
        plt.title("Marker Weights (Regression Coefficients)")
        plt.xlabel("Marker index")
        plt.ylabel("Weight")
        plt.show()

    def predicted_vs_actual_training_population(self):
        y_pred = self.model.predict(self.population._markers.values)
        plt.scatter(self.target_phenotype, y_pred)
        plt.xlabel("Actual phenotype")
        plt.ylabel("Predicted phenotype")
        plt.title("Predicted vs Actual (Training)")
        plt.plot([self.target_phenotype.min(), self.target_phenotype.max()],
                 [self.target_phenotype.min(), self.target_phenotype.max()], 'r--')
        plt.show()

    def predict_phenotypes(self, population):
        return self.model.predict(population._markers.values)

def get_biotic_stress_grn():
    return GRNModel()


class GRNModel:
    def __init__(self, params=None):
        self.params = params or {
            "k_act": 2.0,
            "K": 1.0,
            "n": 2,
            "gamma": 1.0,
        }
        self.knockouts = set()
        self.genes = ["ABA", "ABF2", "ANAC019", "GENEC", "ICS1", "SA", "BGL2"]
        self.network = [
            ("ABA", "ABF2", "activates"),
            ("ABA", "GENEC", "activates"),
            ("ABF2", "ANAC019", "activates"),
            ("ANAC019", "ICS1", "represses"),
            ("GENEC", "ICS1", "represses"),
            ("ICS1", "SA", "activates"),
            ("SA", "BGL2", "activates"),
        ]
        # Give each interaction a random strength modifier (to make them a bit different)
        rng = np.random.default_rng()
        self.strength_modififiers = rng.uniform(0.5,1.5,len(self.network))

    def hill_activation(self, x):
        p = self.params
        return (x ** p["n"]) / (p["K"] ** p["n"] + x ** p["n"])

    def hill_repression(self, x):
        p = self.params
        return (p["K"] ** p["n"]) / (p["K"] ** p["n"] + x ** p["n"])

    def ode_system(self, y, t, stress_func):

        p = self.params
        gene_idx = {gene: i for i, gene in enumerate(self.genes)}
        dydt = [0.0] * len(self.genes)

        # External input: Drought → ABA
        drought = stress_func(t)
        if "ABA" in self.knockouts:
            dydt[gene_idx["ABA"]] = -p["gamma"] * y[gene_idx["ABA"]]
        else:
            dydt[gene_idx["ABA"]] = p["k_act"] * drought - p["gamma"] * y[gene_idx["ABA"]]

        # Handle ICS1 separately with AND logic for its repressors
        if "ICS1" not in self.knockouts:
            anac_val = y[gene_idx["ANAC019"]]
            genec_val = y[gene_idx["GENEC"]]
            rep_anac = self.hill_repression(anac_val)
            rep_genec = self.hill_repression(genec_val)
            and_repression = rep_anac * rep_genec
            dydt[gene_idx["ICS1"]] += self.strength_modififiers[4] * p["k_act"] * and_repression
        else:
            dydt[gene_idx["ICS1"]] = -p["gamma"] * y[gene_idx["ICS1"]]

        # Now apply regular logic for other interactions (excluding ICS1)
        for i, (src, tgt, interaction) in enumerate(self.network):
            if tgt == "ICS1":
                continue  # already handled
            if tgt in self.knockouts:
                dydt[gene_idx[tgt]] = -p["gamma"] * y[gene_idx[tgt]]
                continue
            src_val = y[gene_idx[src]]
            effect = (
                self.hill_activation(src_val) if interaction == "activates"
                else self.hill_repression(src_val)
            )
            dydt[gene_idx[tgt]] += self.strength_modififiers[i] * p["k_act"] * effect


        # Apply degradation
        for i, gene in enumerate(self.genes):
            dydt[i] -= p["gamma"] * y[i]

        return dydt

    def simulate(self, t=None, y0=None, stress_func=None):
        if t is None:
            t = np.linspace(0, 50, 500)
        if y0 is None:
            y0 = [0.1] * len(self.genes)
        if stress_func is None:
            stress_func = lambda t: 0  

        sol = odeint(self.ode_system, y0, t, args=(stress_func,))
        self.t = t
        self.sol = sol
        self.plot()

    def plot(self):
        if not hasattr(self, "sol"):
            raise RuntimeError("Run .simulate() first.")
        for i, gene in enumerate(self.genes):
            plt.plot(self.t, self.sol[:, i], label=gene)
        plt.xlabel("Time")
        plt.ylabel("Expression")
        plt.title("Gene Expression Dynamics")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def knock_out(self, gene):
        if gene not in self.genes:
            raise ValueError(f"{gene} not in gene list.")
        self.knockouts.add(gene)

    def reset_knockouts(self):
        self.knockouts.clear()

    def plot_network(self):
        G = nx.DiGraph()
        for src, tgt, interaction in self.network:
            G.add_edge(src, tgt, interaction=interaction)
        G.add_node("Drought")
        G.add_edge("Drought", "ABA", interaction="activates")

        pos = nx.spectral_layout(G)
        plt.figure(figsize=(7, 5))
        for src, tgt in G.edges:
            interaction = G[src][tgt]["interaction"]
            style = "solid" if interaction == "activates" else "dashed"
            colour = "green" if interaction == "activates" else "red"
            nx.draw_networkx_edges(G, pos, edgelist=[(src, tgt)],
                                   edge_color=colour, style=style,
                                   arrows=True, arrowstyle='-|>')
        nx.draw_networkx_nodes(G, pos, alpha=.5, node_size=300)
        nx.draw_networkx_labels(G, pos)
        plt.title("Gene Regulatory Network")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

def train_dl_model(train_dataset, validation_dataset, epochs=2, batch_size=32):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        tf.keras.layers.MaxPooling2D(4, 4),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid') # Sigmoid for binary classification
    ])
    model.compile(loss='binary_crossentropy',
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])
    model.summary()

    batched_train = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    batched_val = validation_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    history = model.fit(
        batched_train,
        epochs=epochs,
        validation_data=batched_val
    )
    return model