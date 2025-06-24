from dsplantbreeding.Population import get_agricultural_population, get_natural_population, get_resilient_population
from dsplantbreeding.actions import perform_cross_between

def test_gwas():
    # MAS

    # Load toy dataset of natural variation.
    my_population = get_natural_population()
    my_population.show_manhattan_plot('Salt Resistance')
    print()
    # Do GWAS on this to identify SNPs for resilience to some stress.

    # Now we pick a high-yield plant without resilience (and some plant with resilience (not important here how we pick this one))

    # We have two plants, one has resilience, the other has high yield.

    # We perform crosses between them and measure markers.

    # Now we end up with a population of plants that have high yield and resilience.

def test_crosses():
    
    resilient_population = get_resilient_population()
    # Now check if it indeed contains the snps
    resilient_population.show_snps_at_location('SNP_12')

    agricultural_population = get_agricultural_population()

    # check difference in phenotypes
    resilient_population.show_phenotypes()
    agricultural_population.show_phenotypes()
    
    new_population = perform_cross_between(resilient_population, agricultural_population, n_offspring=10)

    selected_population = new_population.select_plants_with_snp_at_location(12, desired_allele=1)
    selected_population.n_plants
    backcross_1 = perform_cross_between(selected_population, agricultural_population, n_offspring=10)

    selected_back1_population = backcross_1.select_plants_with_snp_at_location(12, desired_allele=1)

    backcross_2 = perform_cross_between(selected_back1_population, agricultural_population, n_offspring=10)
    selected_back2_population = backcross_2.select_plants_with_snp_at_location(12, desired_allele=1)
    selected_back2_population.show_phenotypes()