'''Code to predict structures and produce pdbs of the full antibody with and without relaxing
the structures'''

from ABDB import database as db
import matplotlib.pyplot as plt
import numpy as np
from retrain_ablooper import *
from retrain_ablooper.format_outputs import produce_full_structures_of_val_set
import torch
import pandas as pd
from rich import print as pprint
from ABlooper import CDR_Predictor

# torch settings
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float)

model = MaskDecoyGen(decoys=5).to(device = device).float()
model.load_state_dict(torch.load('best_models/best_model-2804-Radam-5-2optim', map_location=torch.device(device)))

batch_size = 1

train = torch.load('train_data/train.pt')
validation = torch.load('train_data/val.pt')
test = torch.load('train_data/test.pt')

val_dataloader = torch.utils.data.DataLoader(validation, 
                                             batch_size=batch_size,
                                             num_workers=1,
                                             shuffle=False,
                                             pin_memory=True,
                                             )

test_dataloader = torch.utils.data.DataLoader(test, 
                                              batch_size=batch_size,
                                              num_workers=1,
                                              shuffle=False,
                                              pin_memory=True,
                                              )

model = MaskDecoyGen(decoys=5).to(device = device).float()
model.load_state_dict(torch.load('best_models/best_model-2804-Radam-5-2optim', map_location=torch.device(device)))

# for i in range(5):
#     tmp = MaskDecoyGen(decoys=1).to(device = device).float()
#     dict=torch.load("best_models/best_model-0305-Radam-1-2optim-"+str(i+1), map_location=torch.device(device))
#     tmp.load_state_dict(dict)
#     weights = tmp.blocks[0].state_dict()
# 
#     model.blocks[i].load_state_dict(weights)
# 

print('predict test set')
pdb_ids, CDR_rmsds_not_relaxed, CDR_rmsds_relaxed, decoy_diversities, CDR_rmsds_not_relaxed_recalc = produce_full_structures_of_val_set(test_dataloader, model, outdir='2804-Radam-5-2optim-test', relax=True)

with open('pdbs/2804-Radam-5-2optim-test/metrics.json', 'w') as f:
    json.dump({'pdb_ids': pdb_ids, 'cdr_rmsds': CDR_rmsds_not_relaxed, 'cdr_rmsds_relaxed': CDR_rmsds_relaxed, 'decoy_divsersity': decoy_diversities, 'cdr_rmsds_recalc': CDR_rmsds_not_relaxed_recalc}, f)

print('predict val set')
pdb_ids, CDR_rmsds_not_relaxed, CDR_rmsds_relaxed, decoy_diversities, CDR_rmsds_not_relaxed_recalc = produce_full_structures_of_val_set(val_dataloader, model, outdir='2804-Radam-5-2optim', relax=True)

with open('pdbs/2804-Radam-5-2optim/metrics.json', 'w') as f:
    json.dump({'pdb_ids': pdb_ids, 'cdr_rmsds': CDR_rmsds_not_relaxed, 'cdr_rmsds_relaxed': CDR_rmsds_relaxed, 'decoy_divsersity': decoy_diversities, 'cdr_rmsds_recalc': CDR_rmsds_not_relaxed_recalc}, f)

