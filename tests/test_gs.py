from dsplantbreeding.Population import get_natural_population
from dsplantbreeding.actions import perform_cross_between
import numpy as np

def test_gs():
    natural_population = get_natural_population(n_plants=50, n_markers=50)
    genomic_selection_model = natural_population.fit_gs_model(target_phenotype='Yield')
    predicted_values = genomic_selection_model.predict_phenotypes(natural_population) # Also called GEBV
    assert np.allclose(predicted_values, [10.43664289,  7.71680075, 12.4261732 ,  6.80288849, 10.2812453 ,
        7.11047349, 11.80468667,  5.26393781,  9.2554396 ,  9.33370826,
        6.9222161 ,  3.25730484,  9.47505364,  6.21556983, 14.0077378 ,
        3.63492133,  5.2782573 ,  7.92555069,  5.91273909, 13.70426704,
        9.2043126 ,  8.98923647, 11.20700599,  7.61039131,  4.47338663,
       10.44542444,  4.41436128,  6.07680405,  1.89587991,  8.49877246,
       14.28670458,  6.29697294, 10.99411354, 10.1005182 ,  8.31770101,
       12.31176206, 11.1513988 ,  7.06346787, 10.99407172,  8.69579883,
        9.87206357,  6.17925449,  7.25143564, 14.23127427,  7.45554059,
        7.27609224, 10.78507141,  6.46134501,  3.84449086, 13.31714103])
    new_population = natural_population.select_plants_from_predicted_values(predicted_values, ntop=2)
    assert new_population.n_plants == 2, "Expected 2 plants in the new population"