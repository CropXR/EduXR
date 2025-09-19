
import pandas as pd

from goatools.obo_parser import GODag
from goatools.go_enrichment import GOEnrichmentStudy
from goatools.anno.gaf_reader import GafReader
from goatools.evidence_codes import EvidenceCodes

from goatools.godag_plot import plot_gos
from IPython.display import Image, display
import tempfile

def show_annotations_of_gene(gene_id: str, tair_gaf: GafReader, godag: GODag):
    annotations = tair_gaf.get_id2gos('BP')  
    nts = tair_gaf.associations
    assert gene_id in annotations, f"Gene {gene_id} not found in annotations."

    # Print hierarchy
    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
        filename = tmp.name

        plot_gos(filename, 
            list(annotations[gene_id]), # Source GO ids
            godag)

        # Display it in the notebook
        display(Image(filename))

    for nt in nts:
        if nt.DB_ID == gene_id and nt.GO_ID in annotations[gene_id]:
            print(f' {nt.GO_ID}, Source: {nt.Assigned_By}, {nt.Evidence_Code}, {EvidenceCodes.code2nt[nt.Evidence_Code].name}')

def print_all_evidence_codes():
    for i, j in EvidenceCodes.code2nt.items():
        print(i, j.name)

def perform_go_enrichment(clustered_df, cluster_id, tair_gaf, godag, fdr_cutoff=0.05):
    goeaobj = GOEnrichmentStudy(
        clustered_df.index.tolist(),  
        tair_gaf.get_id2gos('BP'),
        godag,
        methods=['fdr_bh'],
        )
    results = goeaobj.run_study(clustered_df.groupby('cluster').groups[cluster_id].tolist(), prt=None)
    rows = []
    for ntd in results:
        if ntd.p_fdr_bh < fdr_cutoff:
            rows.append({
                'NS': ntd.NS,
                'GO': ntd.GO,
                'Name': ntd.name,
                'e/p': ntd.enrichment,
                'pval_uncorr': ntd.p_uncorrected,
                'BH': ntd.p_fdr_bh,
                'study_ratio': f"{ntd.ratio_in_study[0]}/{ntd.ratio_in_study[1]}",
                'pop_ratio': f"{ntd.ratio_in_pop[0]}/{ntd.ratio_in_pop[1]}"
            })

    if not rows:
        print('\nNo Enriched GO terms found')
        return None
    df_results = pd.DataFrame(rows)
    df_results = df_results.sort_values(['BH'])
    
    # Make a temporary file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
        filename = tmp.name

        plot_gos(filename, 
            df_results['GO'], # Source GO ids
            godag, 
            goea_results=results)

        # Display it in the notebook
        display(Image(filename))

    return df_results