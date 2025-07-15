from dsplantbreeding.Models import get_biotic_stress_grn
from dsplantbreeding.Datasets.TimeSeriesTranscriptomics import TimeSeriesTranscriptomics
import pytest
import matplotlib.pyplot as plt

@pytest.fixture
def my_grn():
    return get_biotic_stress_grn()


def test_get_n_genes(my_grn):
    assert my_grn.genes == ["ABA", "ABF2", "ANAC019", "GENEC", "ICS1", "SA", "BGL2"]


def test_plot_grn_topology(my_grn):
    my_grn.plot_network()
    


def test_plot_grn_expressions(my_grn):
    my_grn.simulate()


def test_knockout_gene(my_grn):
    my_grn.knock_out("GENEC")
    my_grn.knockouts == {'GENEC'}
