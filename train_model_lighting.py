import numpy as np
from rich.progress import track
import copy
from sklearn.model_selection import train_test_split
import torch
from einops import rearrange
import json
import pandas as pd
import pytorch_lightning
from pytorch_lightning.loggers.neptune import NeptuneLogger


# 1. Functions to reformat data to input
aa1 = "ACDEFGHIKLMNPQRSTVWY"
aa3 = ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU", "MET", "ASN", "PRO", "GLN", "ARG", "SER",
       "THR", "VAL", "TRP", "TYR", ]

short2long = {}
long2short = {}
short2num = {}

for ind in range(0, 20):
    long2short[aa3[ind]] = aa1[ind]
    short2long[aa1[ind]] = aa3[ind]
    short2num[aa1[ind]] = ind

def encode(x, classes):
    '''
    One hot encodes a scalar x into a vector of length classes.
    This is the function used for Sequence encoding.
    '''
    one_hot = np.zeros(classes)
    one_hot[x] = 1

    return one_hot

def one_hot(num_list, classes=20):
    '''
    One hot encodes a 1D vector x.
    This is the function used for Sequence encoding.
    '''
    end_shape = (len(num_list), classes)
    finish = np.zeros(end_shape)
    for i in range(end_shape[0]):
        finish[i] = encode(num_list[i], classes)

    return finish

def which_loop(loop_seq, cdr):
    '''
    Adds a one-hot encoded vector to each node describing which CDR it belongs to.
    '''
    CDRs = ["H1", "H2", "H3", "L1", "L2", "L3", "Anchor"]
    loop = np.zeros((len(loop_seq), len(CDRs)))
    loop[:, -1] = 1
    loop[2:-2] = np.array([1.0 if cdr == x else 0.0 for x in (CDRs)])[None].repeat(len(loop_seq) - 4, axis=0)

    return loop

def positional_encoding(sequence, n=5):
    '''
    Gives the network information on how close each resdiue is to the anchors
    '''
    encs = []
    L = len(sequence)
    for i in range(n):
        encs.append(np.cos((2 ** i) * np.pi * np.arange(L) / L))
        encs.append(np.sin((2 ** i) * np.pi * np.arange(L) / L))

    return np.array(encs).transpose()

def res_to_atom(res_encoding, n_atoms=4):
    '''
    Adds a one-hot encoded vector to each node describing what atom type it is.
    '''
    out_shape = (res_encoding.shape[0], n_atoms, 41)
    atom_encoding = np.zeros(out_shape)

    for i in range(len(res_encoding)):
        for j in range(n_atoms):
            atom_encoding[i, j, 0:37] = res_encoding[i]
            # add one-hot encoding for atom type
            atom_encoding[i, j, 37:] = one_hot([j], classes=n_atoms) 

    return atom_encoding

def prepare_input_loop(CDR_coord, CDR_seq, CDR):
    '''
    Generates input features to be fed into the network for a single CDR
    '''
    CDR_input_coord = copy.deepcopy(CDR_coord)
    # put CDR residues equally spaced on straight line between anchor residues 
    CDR_input_coord[1:-1] = np.linspace(CDR_coord[1], CDR_coord[-2], len(CDR_coord) - 2)
    # CDR_input_coord = rearrange(torch.tensor(CDR_input_coords), "i a d -> () (i a) d").float()

    one_hot_encoding = one_hot(np.array([short2num[amino] for amino in CDR_seq]))
    loop = which_loop(CDR_seq, CDR)
    positional = positional_encoding(CDR_seq)
    res_encoding = np.concatenate([one_hot_encoding, positional, loop], axis=1)
    atom_encoding = res_to_atom(res_encoding)

    # encoding = res_to_atom(torch.tensor(np.concatenate([one_hot_encoding, positional, loop], axis=1)).float())
    # encoding = rearrange(encoding, "i a d -> () (i a) d")

    return CDR_input_coord, atom_encoding

def prepare_model_input(CDR_seq, CDR_BB_coord):
    '''
    Prepares model inputs for a single FAB
    '''
    encodings = []
    geomins = []
    
    for CDR in CDR_BB_coord:
        geom, encode = prepare_input_loop(CDR_BB_coord[CDR], CDR_seq[CDR], CDR)
        encodings.append(encode)
        geomins.append(geom)

    # concatenate encodings and geoms into single array
    encodings = np.concatenate(encodings, axis=0)
    geomins = np.concatenate(geomins, axis=0)
    # format to tensor
    encodings = torch.from_numpy(encodings)
    geomins = torch.from_numpy(geomins)
    # rearrange tensors that atoms in one residue are nolonger grouped
    encodings = rearrange(encodings, "i a d -> (i a) d")
    geomins = rearrange(geomins, "i a d -> (i a) d")

    return geomins, encodings

def prepare_model_inputs(CDR_seqs, CDR_BB_coords):
    '''
    Prepares model inputs for a list of FABs
    '''
    encodings = []
    geomins = []

    for i in track(range(len(CDR_seqs)), description='Preparing model inputs'):
        geom, encode = prepare_model_input(CDR_seqs[i], CDR_BB_coords[i])
        encodings.append(encode)
        geomins.append(geom)

    return geomins, encodings

def prepare_model_output(CDR_BB_coords):
    '''
    Prepares model outputs for training, formated identically to inputs
    '''
    geomouts = []
    for CDR_BB_coord in track(CDR_BB_coords, description='Preparing model outputs'):
        geomout = []
        for _, coords in CDR_BB_coord.items():
            geomout.append(coords)

        # concatenate geoms into single array
        geomout = np.concatenate(geomout, axis=0)
        # format to tensor
        geomout = torch.from_numpy(geomout)
        # rearrange tensor
        geomout = rearrange(geomout, "i a d -> (i a) d")

        geomouts.append(geomout)
    return geomouts

def concatenate_data(encodings, geomins, geomouts, masks):
    '''
    Puts encodings, geomins and geomouts into a single array.
    '''
    data = []
    for i in range(len(encodings)):
        # potentially change list to dict
        data.append({'encodings': encodings[i],
                     'geomins': geomins[i],
                     'geomouts': geomouts[i],
                     'mask': masks[i]})

    return data

# creat mask
def create_mask(node_encodings, mask_lenght=504):
    '''
    Function that creates a mask for each element in a array. Mask is of length <mask_lenght> 
    with 1s where there is an atom and filled up with 0s
    '''
    masks = []
    for node_encoding in node_encodings:
        mask = torch.zeros((mask_lenght))
        mask[:len(node_encoding)] = 1
        masks.append(mask)

    return masks
    
# padding
def pad_tensor(tensor, out_lenght=504):
    '''
    Pads all tensors with 0 to the same number of nodes
    '''
    ndim = tensor.shape[1]
    outtensor = torch.zeros((out_lenght, ndim))
    outtensor[:tensor.shape[0], :] = tensor
    return outtensor

def pad_list_of_tensors(tensors, pad_length=504):
    '''
    Pads all tensors in a list
    '''
    tensors_padded = []

    for i in range(len(tensors)):
        tensors_padded.append(pad_tensor(tensors[i], out_lenght=pad_length))

    return tensors_padded



# Implement model
class MaskEGNN(torch.nn.Module):
    '''
    Singel layer of an EGNN.
    '''
    def __init__(self, node_dim, message_dim=32):
        super().__init__()

        edge_input_dim = (node_dim * 2) + 1

        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(edge_input_dim, 2*edge_input_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(2*edge_input_dim, message_dim),
            torch.nn.SiLU()
        )

        self.node_mlp = torch.nn.Sequential(
            torch.nn.Linear(node_dim + message_dim, 2*node_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(2*node_dim, node_dim),
        )

        self.coors_mlp = torch.nn.Sequential(
            torch.nn.Linear(message_dim, 4*message_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(4*message_dim, 1)
        )
    
    def forward(self, node_features, coordinates, mask):                                                        # We pass in a mask that tells us what nodes to consider and which to ignore.
        pair_mask = rearrange(mask, 'b j -> b () j ()') * rearrange(mask, 'b i -> b i () ()')
        
        rel_coors = rearrange(coordinates, 'b i d -> b i () d') - rearrange(coordinates, 'b j d -> b () j d')  
        rel_dist = (rel_coors ** 2).sum(dim=-1, keepdim=True)                                                  

        feats_j = rearrange(node_features, 'b j d -> b () j d')      
        feats_i = rearrange(node_features, 'b i d -> b i () d')
        feats_i, feats_j = torch.broadcast_tensors(feats_i, feats_j)

        edge_input = torch.cat((feats_i, feats_j, rel_dist), dim=-1)

        m_ij = self.edge_mlp(edge_input)

        coor_weights = pair_mask * self.coors_mlp(m_ij)                                                          # We multiply the predicted weight by the mask (masked residue pairs will have zero weight).
        coor_weights = rearrange(coor_weights, 'b i j () -> b i j')

        rel_coors_normed = rel_coors / rel_dist.clip(min = 1e-8)    

        coors_out = coordinates + torch.einsum('b i j, b i j c -> b i c', coor_weights, rel_coors_normed)  

        m_i = torch.einsum('b i d, b -> b i d', (pair_mask * m_ij).sum(dim=-2), 1/mask.sum(-1))                 # To average we divide over the length for each batch (length = sum(mask)).

        node_mlp_input = torch.cat((node_features, m_i), dim=-1)
        node_out = node_features + mask.unsqueeze(-1) * self.node_mlp(node_mlp_input)                             # We set the update for maked residues to zero. 

        return node_out, coors_out

class MaskEGNNModel(torch.nn.Module):
    '''
    4 EGNN layers joined into one Model
    '''
    def __init__(self, node_dim, layers=4, message_dim=32):
        super().__init__()

        self.layers = torch.nn.ModuleList([MaskEGNN(node_dim, message_dim = message_dim) for _ in range(layers)])   # Initialise as many EGNN layers as needed

    def forward(self, node_features, coordinates, mask):

        for layer in self.layers:                                                                            
            node_features, coordinates = layer(node_features, coordinates, mask)                                      # Update node features and coordinates for each layer in the model
        
        return node_features, coordinates

class MaskDecoyGen(torch.nn.Module):
    '''
    5 EGNN models run in parallel.
    '''
    def __init__(self, dims_in=41, decoys=5, **kwargs):
        super().__init__()
        self.blocks = torch.nn.ModuleList([MaskEGNNModel(node_dim=dims_in, **kwargs) for _ in range(decoys)])
        self.decoys = decoys

    def forward(self, node_features, coordinates, mask):
        geoms = torch.zeros((self.decoys, *coordinates.shape[1:]), device=coordinates.device)

        for i, block in enumerate(self.blocks):
            geoms[i] = block(node_features, coordinates, mask)[1] # only save geoms

        return geoms



# loss functions
def rmsd(prediction, truth):
    dists = (prediction - truth).pow(2).sum(-1)
    return torch.sqrt(dists.mean(-1)).mean()

def rmsds(preds, true):
    return  torch.sort((preds - true).pow(2).sum(-1).mean(-1).pow(1/2))[0]

def length_penalty(pred):
    return ((((pred[:,1:]-pred[:,:-1])**2).sum(-1).pow(1/2) - 3.802).pow(2)).mean()

def different_penalty(pred):
    return -(rearrange(pred, "i n d -> i () n d") - rearrange(pred, "j n d -> () j n d")).pow(2).mean()

def dist_check(pred, amino):
    err = 0
    for i in range(6):
        CDR = rearrange(pred[:,amino[0,:,30+i]==1.0], "d (r a) p -> d a r p", a = 4)
        # CA-CA
        err += (((CDR[:,0,1:] - CDR[:,0,:-1]).pow(2).sum(-1).pow(1/2) - 3.82).abs() - 0.12).clamp(0).mean()
        # CA-N
        err += (((CDR[:,0] - CDR[:,1]).pow(2).sum(-1).pow(1/2) - 1.47).abs() - 0.01).clamp(0).mean()
        # CA-C
        err += (((CDR[:,0] - CDR[:,2]).pow(2).sum(-1).pow(1/2) - 1.53).abs() - 0.01).clamp(0).mean()
        # C-N
        err += (((CDR[:,2,:-1] - CDR[:,1,1:]).pow(2).sum(-1).pow(1/2) - 1.34).abs() - 0.01).clamp(0).mean()
        # CA-CB
        CDR2 = rearrange(pred[:,(amino[0,:,30+i]==1.0) & (amino[0,:,5] != 1.0)], "d (r a) p -> d a r p", a = 4)
        err += (((CDR2[:,0] - CDR2[:,-1]).pow(2).sum(-1).pow(1/2) - 1.54).abs() - 0.01).clamp(0).mean()

    return err

def atom_dist(geom):
    return ((geom[:,None] - geom[:,:,None]).mean(0).pow(2).sum(-1) + 1e-8).pow(1/2) 

def atom_dist_penal(geom, pred):
    true_ds = atom_dist(geom)
    pred_ds = atom_dist(pred)
    mask = true_ds < 4.0
    return (true_ds-pred_ds)[mask].pow(2).mean()




# lighting model
class pl_EGNNModel(pytorch_lightning.LightningModule):
    def __init__(self):
        super().__init__()
        self.egnnmodel = MaskDecoyGen()

    def forward(self, node_encodings, coordinates, mask):

        return self.egnnmodel(node_encodings, coordinates, mask)   

    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(model.parameters(), lr=1e-3, weight_decay=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        predicted_coordinates = self(batch['encodings'], batch['geomins'], batch['mask'])  
        loss = rmsd(batch['geomouts'], predicted_coordinates)
        return loss

    def validation_step(self, batch, batch_idx): 
        predicted_coordinates = self(batch['encodings'], batch['geomins'], batch['mask'])
        loss = rmsd(batch['geomouts'],  predicted_coordinates)
        return loss

    def validation_epoch_end(self, val_step_outputs): # Updated once when validation is called
        val_loss = torch.stack(val_step_outputs).detach().cpu().numpy().mean()
        self.logger.experiment['evaluation/val_loss'].log(val_loss)


# import data
with open('train_data/CDR_BB_coords.npy', 'rb') as infile:
    CDR_BB_coords = np.load(infile, allow_pickle=True)

with open('train_data/CDR_seqs.npy', 'rb') as infile:
    CDR_seqs = np.load(infile, allow_pickle=True)

with open('train_data/CDR_BB_coords_test.npy', 'rb') as infile:
    CDR_BB_coords_test = np.load(infile, allow_pickle=True)

with open('train_data/CDR_seqs_test.npy', 'rb') as infile:
    CDR_seqs_test = np.load(infile, allow_pickle=True)

# format data
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

# split in train and validation sets
train, validation = train_test_split(data, test_size=100, random_state=42)

# data loaders
batch_size = 32
train_dataloader = torch.utils.data.DataLoader(train, 
                                               batch_size=batch_size,   # Batch size
                                               num_workers=4,           # Number of cpu's allocated to load the data (recommended is 4/GPU)
                                               shuffle=True,            # Whether to randomly shuffle data
                                               pin_memory=True,         # Enables faster data transfer to CUDA-enabled GPUs (page-locked memory)
                                               )

val_dataloader = torch.utils.data.DataLoader(validation, 
                                             batch_size=batch_size,
                                             num_workers=4,
                                             shuffle=True,
                                             pin_memory=True,
                                             )

test_dataloader = torch.utils.data.DataLoader(test, 
                                              batch_size=batch_size,
                                              num_workers=4,
                                              shuffle=True,
                                              pin_memory=True,
                                              )

# train model
# torch settings
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float)

ourlogger = NeptuneLogger(api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyMGI0ZTUzYy0zMTBkLTRjMWMtODhjNS0wNTJmNjA1MzhmOGMifQ==",
              project="fspoendlin/ABlooper",
              name="Fabian",
              log_model_checkpoints=False,
              )

trainer = pytorch_lightning.Trainer(
    accelerator="auto",  # 'cpu' or 'gpu'
    max_epochs=5000,
    check_val_every_n_epoch=1,
    accumulate_grad_batches=None,
    gradient_clip_val=1.0,
    logger=ourlogger,
    )

model = pl_EGNNModel()
trainer.fit(model, train_dataloader, val_dataloader)
