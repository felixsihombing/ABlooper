'''Implementation of EGNN model. There are two implementations provided, one that
uses a mask to allow batching and one without.'''

import torch
from einops import rearrange
# import pytorch_lightning
# from pytorch_lightning.loggers.neptune import NeptuneLogger

class EGNN(torch.nn.Module):
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
            torch.nn.Linear(message_dim, 2*message_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(2*message_dim, 1)
        )
    
    def forward(self, node_features, coordinates):                                                        # We pass in a mask that tells us what nodes to consider and which to ignore.
        rel_coors = rearrange(coordinates, 'b i d -> b i () d') - rearrange(coordinates, 'b j d -> b () j d')  
        rel_dist = (rel_coors ** 2).sum(dim=-1, keepdim=True)                                                  

        feats_j = rearrange(node_features, 'b j d -> b () j d')      
        feats_i = rearrange(node_features, 'b i d -> b i () d')
        feats_i, feats_j = torch.broadcast_tensors(feats_i, feats_j)

        edge_input = torch.cat((feats_i, feats_j, rel_dist), dim=-1)

        m_ij = self.edge_mlp(edge_input)

        coor_weights = self.coors_mlp(m_ij)                                                          # We multiply the predicted weight by the mask (masked residue pairs will have zero weight).
        coor_weights = rearrange(coor_weights, 'b i j () -> b i j')

        rel_coors_normed = rel_coors / rel_dist.clip(min = 1e-8)    

        coors_out = coordinates + torch.einsum('b i j, b i j c -> b i c', coor_weights, rel_coors_normed)  

        m_i = m_ij.sum(dim=-2)                                                                      # To average we divide over the length for each batch (length = sum(mask)).

        node_mlp_input = torch.cat((node_features, m_i), dim=-1)
        node_out = node_features + self.node_mlp(node_mlp_input)                            # We set the update for maked residues to zero. 

        return node_out, coors_out

class EGNNModel(torch.nn.Module):
    '''
    4 EGNN layers joined into one Model
    '''
    def __init__(self, node_dim, layers=4, message_dim=32):
        super().__init__()

        self.layers = torch.nn.ModuleList([EGNN(node_dim, message_dim = message_dim) for _ in range(layers)])   # Initialise as many EGNN layers as needed

    def forward(self, node_features, coordinates):

        for layer in self.layers:                                                                            
            node_features, coordinates = layer(node_features, coordinates)                                      # Update node features and coordinates for each layer in the model
        
        return node_features, coordinates

class DecoyGen(torch.nn.Module):
    '''
    5 EGNN models run in parallel.
    '''
    def __init__(self, dims_in=41, decoys=1, **kwargs):
        super().__init__()
        self.blocks = torch.nn.ModuleList([EGNNModel(node_dim=dims_in, **kwargs) for _ in range(decoys)])
        self.decoys = decoys

    def forward(self, node_features, coordinates):
        geoms = torch.zeros((self.decoys, *coordinates.shape[1:]), device=coordinates.device)

        for i, block in enumerate(self.blocks):
            geoms[i] = block(node_features, coordinates)[1] # only save geoms

        return geoms

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

# model to use with pytorch lightning
# class pl_EGNNModel(pytorch_lightning.LightningModule):
#     def __init__(self):
#         super().__init__()
#         self.egnnmodel = MaskDecoyGen()
# 
#     def forward(self, node_encodings, coordinates, mask):
# 
#         return self.egnnmodel(node_encodings, coordinates, mask)   
# 
#     def configure_optimizers(self):
#         optimizer = torch.optim.RAdam(model.parameters(), lr=1e-3, weight_decay=1e-3)
#         return optimizer
# 
#     def training_step(self, batch, batch_idx):
#         predicted_coordinates = self(batch['encodings'], batch['geomins'], batch['mask'])  
#         loss = rmsd(batch['geomouts'], predicted_coordinates)
#         return loss
# 
#     def validation_step(self, batch, batch_idx): 
#         predicted_coordinates = self(batch['encodings'], batch['geomins'], batch['mask'])
#         loss = rmsd(batch['geomouts'],  predicted_coordinates)
#         return loss
# 
#     def validation_epoch_end(self, val_step_outputs): # Updated once when validation is called
#         val_loss = torch.stack(val_step_outputs).detach().cpu().numpy().mean()
#         self.logger.experiment['evaluation/val_loss'].log(val_loss)
