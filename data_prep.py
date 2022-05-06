from ABDB import database as db
import numpy as np
from rich.progress import track
from sklearn.model_selection import train_test_split
from einops import rearrange
import json
import pandas as pd
from retrain_ablooper import *
import torch

# 
# # get pdb ids of test set
# all_ids = pd.read_csv('./train_data/datasets.csv')
# test_ids = all_ids[all_ids['Set'] == 'RAB']
# test_ids = test_ids.PDB_ID.values
# 
# # use imgt numbering
# db.set_numbering_scheme("imgt")
# db.set_region_definition("imgt")
# 
# # list of all pdb ids in SAbDab
# all_pdbs_in_sabdab = list(db.db_summary.keys())
# train_val_set = set(all_pdbs_in_sabdab) - set(test_ids)
# print('train set:', len(train_val_set), 'all pdbs in sabdab set:', len(all_pdbs_in_sabdab), 'test set:', len(test_ids))
# 
# print('load test set')
# CDR_seqs_test, CDR_BB_coords_test, CDR_ids_test = get_sabdab_fabs(test_ids)
# 
# with open('train_data/CDR_BB_coords_test.npy', 'wb') as outfile:
#     np.save(outfile, CDR_BB_coords_test)
# 
# with open('train_data/CDR_seqs_test.npy', 'wb') as outfile:
#     np.save(outfile, CDR_seqs_test)
# 
# with open('train_data/CDR_ids_test.npy', 'wb') as outfile:
#     np.save(outfile, CDR_ids_test)
# 
# 
# print('load training and validation set')
# CDR_seqs, CDR_BB_coords, CDR_ids = get_sabdab_fabs(train_val_set)
# 
# with open('train_data/CDR_BB_coords.npy', 'wb') as outfile:
#     np.save(outfile, CDR_BB_coords)
# 
# with open('train_data/CDR_seqs.npy', 'wb') as outfile:
#     np.save(outfile, CDR_seqs)
# 
# with open('train_data/CDR_ids.npy', 'wb') as outfile:
#     np.save(outfile, CDR_ids)


with open('train_data/CDR_BB_coords.npy', 'rb') as infile:
    CDR_BB_coords = np.load(infile, allow_pickle=True)

with open('train_data/CDR_seqs.npy', 'rb') as infile:
    CDR_seqs = np.load(infile, allow_pickle=True)

with open('train_data/CDR_ids.npy', 'rb') as infile:
    CDR_ids = np.load(infile, allow_pickle=True)

with open('train_data/CDR_BB_coords_test.npy', 'rb') as infile:
    CDR_BB_coords_test = np.load(infile, allow_pickle=True)

with open('train_data/CDR_seqs_test.npy', 'rb') as infile:
    CDR_seqs_test = np.load(infile, allow_pickle=True)

with open('train_data/CDR_ids_test.npy', 'rb') as infile:
    CDR_ids_test = np.load(infile, allow_pickle=True)

CDR_seqs, CDR_BB_coords, CDR_ids = filter_CDR_length(CDR_seqs, CDR_BB_coords, CDR_ids, length_cutoff=22)
CDR_seqs_test, CDR_BB_coords_test, CDR_ids_test = filter_CDR_length(CDR_seqs_test, CDR_BB_coords_test, CDR_ids_test, length_cutoff=22)

CDR_seqs, CDR_BB_coords, CDR_ids = filter_CA_distance(CDR_seqs, CDR_BB_coords, CDR_ids)
CDR_seqs_test, CDR_BB_coords_test, CDR_ids_test = filter_CA_distance(CDR_seqs_test, CDR_BB_coords_test, CDR_ids_test)

CDR_seqs, CDR_BB_coords, CDR_ids = remove_test_set_identities(CDR_seqs, CDR_BB_coords, CDR_ids, CDR_seqs_test)


geomins_test, node_encodings_test = prepare_model_inputs(CDR_seqs_test, CDR_BB_coords_test)
geomouts_test = prepare_model_output(CDR_BB_coords_test)

geomins, node_encodings = prepare_model_inputs(CDR_seqs, CDR_BB_coords)
geomouts = prepare_model_output(CDR_BB_coords)

masks = create_masks(node_encodings)
masks_test = create_masks(node_encodings_test)

# pad all data
node_encodings = pad_list_of_tensors(node_encodings)
geomins = pad_list_of_tensors(geomins)
geomouts = pad_list_of_tensors(geomouts)
node_encodings_test = pad_list_of_tensors(node_encodings_test)
geomins_test = pad_list_of_tensors(geomins_test)
geomouts_test = pad_list_of_tensors(geomouts_test)

# convert from np array with dtype object to list to make compatible with torch
CDR_ids = np.array(CDR_ids).tolist()
CDR_ids_test = np.array(CDR_ids_test).tolist()

data = concatenate_data(node_encodings, geomins, geomouts, masks, CDR_ids)
test = concatenate_data(node_encodings_test, geomins_test, geomouts_test, masks_test, CDR_ids_test)

# split in train and validation sets
train, validation = train_test_split(data, test_size=100, random_state=42)

print('size of sets: train:', len(train), 'val:', len(validation), 'test:', len(test))

torch.save(train, 'train_data/train.pt')
torch.save(validation, 'train_data/val.pt')
torch.save(test, 'train_data/test.pt')
