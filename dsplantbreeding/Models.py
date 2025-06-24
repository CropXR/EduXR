from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


class GenomicSelectionModel:
    def __init__(self, population, target_phenotype: str):
        # Population should be a PlantPopulation instance
        self.population = population
        self.target_phenotype = self.population._phenotypes[target_phenotype]
        self.model = None
        self.fit()
    
    def fit(self):
        X = self.population._snps.values
        y = self.target_phenotype
        self.model = LinearRegression()
        self.model.fit(X, y)

    def show_genomic_selection_snp_weights(self):
        plt.figure(figsize=(10, 4))
        plt.bar(range(len(self.model.coef_)), self.model.coef_)
        plt.title("SNP Weights (Regression Coefficients)")
        plt.xlabel("SNP index")
        plt.ylabel("Weight")
        plt.show()

    def predicted_vs_actual_training_population(self):
        y_pred = self.model.predict(self.population._snps.values)
        plt.scatter(self.target_phenotype, y_pred)
        plt.xlabel("Actual phenotype")
        plt.ylabel("Predicted phenotype")
        plt.title("Predicted vs Actual (Training)")
        plt.plot([self.target_phenotype.min(), self.target_phenotype.max()],
                 [self.target_phenotype.min(), self.target_phenotype.max()], 'r--')
        plt.show()

    def predict_phenotypes(self, population):
        return self.model.predict(population._snps.values)