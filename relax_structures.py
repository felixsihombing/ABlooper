'''Code to predict structures and produce pdbs of the full antibody with and without relaxing
the structures'''

from retrain_ablooper import *
from ABDB import database as db
from ABlooper.utils import filt
from ABlooper.openmm_refine import openmm_refine
from rich import track
import torch
import json

# torch settings
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float)

validation = torch.load('train_data/val.pt')

batch_size = 1
val_dataloader = torch.utils.data.DataLoader(validation, 
                                             batch_size=batch_size,
                                             num_workers=1,
                                             shuffle=False,
                                             pin_memory=True,
                                             )

model = MaskDecoyGen(decoys=5).to(device = device).float()
model.load_state_dict(torch.load('best_models/best_model-2804-Radam-5-2optim', map_location=torch.device(device)))

cdr_rmsds, decoy_diversities, pdb_ids = produce_full_structures_of_val_set(val_dataloader, model, outdir='2804-Radam-5-2optim', relax=True)

with open('pdbs/2804-Radam-5-2optim/metrics.json', 'w') as f:
    json.dump({'pdb_ids': pdb_ids, 'cdr_rmsds': cdr_rmsds, 'decoy_divsersity': decoy_diversities}, f)
