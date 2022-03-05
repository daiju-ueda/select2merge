import os
import pandas as pd
import lib.split2merge as lib
from tqdm import tqdm

l_institutions = ['OCU','Habikino','Kashiwara','Morimoto']

documents_dir = '../documents'
selected_dir = os.path.join(documents_dir, 'selected')
comp_dir = os.path.join(documents_dir, 'complete')
resplit_csvname = 'resplit.csv'
outout_filename = 'source_analysis.csv'

selected = lib.Selected(selected_dir)

df_merged = pd.DataFrame()
for institution in l_institutions:
    target_path = os.path.join(selected_dir, institution, selected[institution], resplit_csvname)
    df_resplitted_institution = pd.read_csv(target_path)
    df_merged = pd.concat([df_merged, df_resplitted_institution])

outpath = os.path.join(comp_dir, outout_filename)
df_merged.to_csv(outpath, index=False)