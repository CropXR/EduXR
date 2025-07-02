from dsplantbreeding.Population import get_agricultural_population, get_natural_population, get_resilient_population
from dsplantbreeding.actions import perform_cross_between
import numpy as np

def test_natural_population(capfd):
    my_population = get_natural_population(n_plants=10)
    assert np.allclose(my_population._markers.iloc[0,:].tolist(), 
                           [1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0])
    my_population.show_size()
    out, err = capfd.readouterr()
    assert out.endswith('Population has 10 plants and 50 markers.\n')
    
    my_population.show_manhattan_plot(to_phenotype='Salt Resistance')

def test_show_phenotpe(capfd):
    resilient_population = get_resilient_population()
    agricultural_population = get_agricultural_population()
    resilient_population.show_all_phenotypes()
    agricultural_population.show_all_phenotypes()
    out, err = capfd.readouterr()
    assert out == '   Salt Resistance     Yield\n0         1.979529  3.959505\n   Salt Resistance     Yield\n0        -0.020471  8.959505\n'