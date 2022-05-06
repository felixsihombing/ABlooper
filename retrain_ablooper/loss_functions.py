'''Loss functions to train the model.'''

from einops import rearrange
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float)


# set loss functions
def rmsd(prediction, truth):
    dists = (prediction - truth).pow(2).sum(-1)
    return torch.sqrt(dists.nanmean(-1)).mean()

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

# functions to get rmsds for each cdr individually
def loop_resi_index(node_features, loop):
    '''
    Returns the indices of node features that belong to the specified cdr
    '''
    loop2num = {'H1': 0,
                'H2': 1,
                'H3': 2,
                'L1': 3,
                'L2': 4,
                'L3': 5,
                'Anchor': 6,
                }
    
    index = loop2num[loop]
    loop_resi_indices = []
    loop_encodings = node_features[:, :, 30:37] # elements 30:37 correspond to loop encodings

    for i in range(len(loop_encodings)): # loop through batches
        loop_resi_index = []
        for j in range(len(loop_encodings[i])): # loop through nodes
            if loop_encodings[i, j, index] == 1:
                loop_resi_index.append(j)
        loop_resi_indices.append(loop_resi_index)
    
    return loop_resi_indices # output of size batch x residues in cdr

def loop_resi_coords(coordinates, node_features, loop):
    '''
    Returns the coordinates of atoms belonging to a specified loop.
    '''    
    resi = torch.zeros_like(coordinates).to(device)
    resi[:,:,:] = torch.nan
    indices = loop_resi_index(node_features, loop)

    for j in range(len(indices)): # loop through batches
        for i in indices[j]: # loop through indices in each batch
            resi[j,i,:] = coordinates[j,i,:]

    return resi # output of size batch x 504 x 3

def rmsd_per_cdr(pred, node_features, out_coordinates, CDRs=["H1", "H2", "H3", "L1", "L2", "L3"]):
    '''
    Calculates the rmsd for a list of CDRs.
    '''
    cdr_rmsd = torch.zeros(len(CDRs)).to(device)
    pred_mean = pred.mean(0) # mean prediction of all decoys in each batch 

    for j in range(len(CDRs)):
        pred_cdr = loop_resi_coords(pred_mean, node_features, CDRs[j])
        true_cdr = loop_resi_coords(out_coordinates, node_features, CDRs[j])
        cdr_rmsd[j] = rmsd(pred_cdr, true_cdr)

    return cdr_rmsd # output of size 6
