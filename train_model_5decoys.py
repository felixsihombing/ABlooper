from matplotlib.pyplot import delaxes
import numpy as np
from rich.progress import track
import copy
import torch
from sklearn.model_selection import train_test_split
from einops import rearrange
import json
import pandas as pd
from retrain_ablooper import *



# import data
with open('train_data/CDR_BB_coords.npy', 'rb') as infile:
    CDR_BB_coords = np.load(infile, allow_pickle=True)

with open('train_data/CDR_seqs.npy', 'rb') as infile:
    CDR_seqs = np.load(infile, allow_pickle=True)

with open('train_data/CDR_BB_coords_test.npy', 'rb') as infile:
    CDR_BB_coords_test = np.load(infile, allow_pickle=True)

with open('train_data/CDR_seqs_test.npy', 'rb') as infile:
    CDR_seqs_test = np.load(infile, allow_pickle=True)


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
len(data), len(test)

print('Data formated')


# split in train and validation sets
train, validation = train_test_split(data, test_size=100, random_state=42)

print(f'Size train set: {len(train)}, val set: {len(validation)}, test set: {len(test)}')

batch_size = 1
num_workers = 1
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


# train model
print('Start training')

# initialise model
model = MaskDecoyGen(decoys=5).to(device = device).float()

# set optimiser
optimiser = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

# Step to actually train the network
train_losses, val_losses = train_model(model, optimiser, train_dataloader, val_dataloader, n_epochs=5000, patience=150, decoys=5)
