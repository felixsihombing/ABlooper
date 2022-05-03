import numpy as np
from retrain_ablooper.training import train_model_2optim
from rich.progress import track
import copy
import torch
from sklearn.model_selection import train_test_split
from einops import rearrange
import json
import pandas as pd
from retrain_ablooper import *



print('Import data')
# import data
with open('train_data/CDR_BB_coords.npy', 'rb') as infile:
    CDR_BB_coords = np.load(infile, allow_pickle=True)

with open('train_data/CDR_seqs.npy', 'rb') as infile:
    CDR_seqs = np.load(infile, allow_pickle=True)

with open('train_data/CDR_BB_coords_test.npy', 'rb') as infile:
    CDR_BB_coords_test = np.load(infile, allow_pickle=True)

with open('train_data/CDR_seqs_test.npy', 'rb') as infile:
    CDR_seqs_test = np.load(infile, allow_pickle=True)

print('Number of fabs after import', len(CDR_seqs))



print('Filter data')
# filter input data
CDR_seqs, CDR_BB_coords = filter_CDR_length(CDR_seqs, CDR_BB_coords, length_cutoff=22)
CDR_seqs_test, CDR_BB_coords_test = filter_CDR_length(CDR_seqs_test, CDR_BB_coords_test, length_cutoff=22)

CDR_seqs, CDR_BB_coords = filter_CA_distance(CDR_seqs, CDR_BB_coords)
CDR_seqs_test, CDR_BB_coords_test = filter_CA_distance(CDR_seqs_test, CDR_BB_coords_test)

CDR_seqs, CDR_BB_coords = remove_test_set_identities(CDR_seqs, CDR_BB_coords, CDR_seqs_test)




print('Format data')
# torch settings
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float)

# prepare model inputs/outputs
geomins, node_encodings = prepare_model_inputs(CDR_seqs, CDR_BB_coords)
geomouts = prepare_model_output(CDR_BB_coords)
geomins_test, node_encodings_test = prepare_model_inputs(CDR_seqs_test, CDR_BB_coords_test)
geomouts_test = prepare_model_output(CDR_BB_coords_test)

masks = create_mask(node_encodings)
masks_test = create_mask(node_encodings_test)

# pad all data
node_encodings = pad_list_of_tensors(node_encodings)
geomins = pad_list_of_tensors(geomins)
geomouts = pad_list_of_tensors(geomouts)
node_encodings_test = pad_list_of_tensors(node_encodings_test)
geomins_test = pad_list_of_tensors(geomins_test)
geomouts_test = pad_list_of_tensors(geomouts_test)

data = concatenate_data(node_encodings, geomins, geomouts, masks)
test = concatenate_data(node_encodings_test, geomins_test, geomouts_test, masks_test)
print('Size train/val set: ', len(data), ', size test set:', len(test))




print('Prepare dataloaders')
# split in train and validation sets
train, validation = train_test_split(data, test_size=100, random_state=42)

print(f'Size train set: {len(train)}, val set: {len(validation)}, test set: {len(test)}')

batch_size = 4
num_workers = 4
train_dataloader = torch.utils.data.DataLoader(train, 
                                               batch_size=batch_size,   # Batch size
                                               num_workers=num_workers,           # Number of cpu's allocated to load the data (recommended is 4/GPU)
                                               shuffle=True,            # Whether to randomly shuffle data
                                               pin_memory=True,         # Enables faster data transfer to CUDA-enabled GPUs (page-locked memory)
                                               )

val_dataloader = torch.utils.data.DataLoader(validation, 
                                             batch_size=batch_size,
                                             num_workers=num_workers,
                                             shuffle=True,
                                             pin_memory=True,
                                             )

test_dataloader = torch.utils.data.DataLoader(test, 
                                              batch_size=batch_size,
                                              num_workers=num_workers,
                                              shuffle=True,
                                              pin_memory=True,
                                              )




print('Start training')
# train model
# initialise model
model = MaskDecoyGen(decoys=1).to(device = device).float()

# set optimisers
optimiser1 = torch.optim.RAdam(model.parameters(), lr=1e-3, weight_decay=1e-3)
optimiser2 = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

# Step to actually train the network
train_losses, val_losses = train_model_2optim(model, optimiser1, optimiser2, train_dataloader, val_dataloader, training_name='-0305-Radam-1-2optim-5' , n_epochs=5000, patience=100, decoys=1)
