from ABDB import database as db
import numpy as np
from rich.progress import track
from sklearn.model_selection import train_test_split
from einops import rearrange
import json
import pandas as pd
from retrain_ablooper import *


# get pdb ids of test set
all_ids = pd.read_csv('./train_data/datasets.csv')
test_ids = all_ids[all_ids['Set'] == 'RAB']
test_ids = test_ids.PDB_ID.values

# use imgt numbering
db.set_numbering_scheme("imgt")
db.set_region_definition("imgt")

# list of all pdb ids in SAbDab
all_pdbs_in_sabdab = list(db.db_summary.keys())
train_val_set = set(all_pdbs_in_sabdab) - set(test_ids)
print('train set:', len(train_val_set), 'all pdbs in sabdab set:', len(all_pdbs_in_sabdab), 'test set:', len(test_ids))

print('load test set')
CDR_seqs_test, CDR_BB_coords_test = get_sabdab_fabs(test_ids)

with open('train_data/CDR_BB_coords_test.npy', 'wb') as outfile:
    np.save(outfile, CDR_BB_coords_test)

with open('train_data/CDR_seqs_test.npy', 'wb') as outfile:
    np.save(outfile, CDR_seqs_test)


print('load training and validation set')
CDR_seqs, CDR_BB_coords = get_sabdab_fabs(train_val_set)

with open('train_data/CDR_BB_coords.npy', 'wb') as outfile:
    np.save(outfile, CDR_BB_coords)

with open('train_data/CDR_seqs.npy', 'wb') as outfile:
    np.save(outfile, CDR_seqs)