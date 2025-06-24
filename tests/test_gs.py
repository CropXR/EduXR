from dsplantbreeding.Population import get_natural_population
from dsplantbreeding.actions import perform_cross_between

def test_gs():
    natural_population = get_natural_population()
    genomic_selection_model = natural_population.fit_gs_model(target_phenotype='Yield')
    genomic_selection_model.show_genomic_selection_snp_weights()
    genomic_selection_model.predicted_vs_actual_training_population()

    predicted_values = genomic_selection_model.predict_phenotypes(natural_population) # Also called GEBV

    natural_population.select_plants_from_predicted_values(predicted_values, ntop=2)

    f1 = perform_cross_between(natural_population, natural_population, n_offspring=10)

    f1_gebv = genomic_selection_model.predict_phenotypes(f1) # Also called GEBV

    selected_f1 = f1.select_plants_from_predicted_values(f1_gebv, ntop=2)

    f2 = perform_cross_between(selected_f1, selected_f1, n_offspring=10)

    # Now select plants based on highest values
    f2.show_phenotypes()
    # This allows us to maximise for traits that are controlled by many genes.

