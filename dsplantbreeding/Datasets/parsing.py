from io import StringIO
import pandas as pd
import tempfile

def biotic_stress_dataset_parsing(in_path, out_path):
    """Some left over code that I used to download and parse the biotic stress dataset. Just saving it here for reference."""
    df = pd.read_excel(in_path, index_col=0)

    df = df.astype(float)
    top_genes = df.var(axis=1).sort_values(ascending=False).head(500).index.tolist()
    df_filtered = df.loc[top_genes]


    data = """ID,Time,Treatment,A/B
Sample_1,1,MOCK,A
Sample_13,1,MOCK,B
Sample_2,1,AVR,A
Sample_14,1,AVR,B
Sample_3,1,VIR,A
Sample_15,1,VIR,B
Sample_7,6,MOCK,A
Sample_19,6,MOCK,B
Sample_8,6,AVR,A
Sample_20,6,AVR,B
Sample_9,6,VIR,A
Sample_21,6,VIR,B
Sample_10,12,MOCK,A
Sample_22,12,MOCK,B
Sample_11,12,AVR,A
Sample_23,12,AVR,B
Sample_12,12,VIR,A
Sample_24,12,VIR,B"""

    metadata = pd.read_csv(StringIO(data))
    metadata['Time_Treatment_AB'] = metadata['Time'].astype(str) + '_' + metadata['Treatment'] + '_' + metadata['A/B']
    metadata = metadata.set_index('ID')

    df_filtered.columns = metadata.loc[df_filtered.columns, 'Time_Treatment_AB']

    df_filtered = df_filtered.T
    df_filtered.index = pd.MultiIndex.from_tuples(df_filtered.index.str.split('_').tolist(), names=['Time', 'Treatment', 'A/B'])
    df_filtered = df_filtered.T
    df_filtered

    df_averaged = df_filtered.groupby(['Time', 'Treatment'], axis=1).mean()
    df_normalized = df_averaged.apply(lambda x: (x - x.mean()) / x.std(), axis=1)

    df_normalized.index = "Gene: " + df_normalized.index
    df_normalized.index.name = None

    df_cond_at_top = df_normalized.T.reset_index().T

    df_cond_at_top.columns = df_cond_at_top.iloc[:2,:].apply(lambda x: f'{x.iloc[1]}_{x.iloc[0]}', axis=0)

    df_cond_at_top.iloc[0] = 'Time: ' + df_cond_at_top.iloc[0].astype(str)
    df_cond_at_top.iloc[1] = 'Treatment: ' + df_cond_at_top.iloc[1].astype(str)
    
    # Write to a temporary file
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.tsv', delete=False) as tmp:
        df_cond_at_top.to_csv(tmp.name, sep='\t', header=True)
        tmp.seek(0)  # Move to beginning to read
        
        lines = tmp.readlines()

    lines[1] = lines[1].replace('Time', '', 1)
    lines[2] = lines[2].replace('Treatment', '', 1)

    # Write the result
    with open(out_path, "w") as f:
        f.writelines(lines)