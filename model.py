import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TransformerConv
from torch_geometric.nn import global_mean_pool
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, node_emb_dim=128, edge_emb_dim=128, n_heads=1):
        super(GCN, self).__init__()
        
        tconv_params = {"heads": n_heads, 
                    "concat" : True, 
                    "beta": False, 
                    "dropout": 0.0, 
                    "edge_dim": edge_emb_dim, 
                    "bias": True, 
                    "root_weight": True}
        torch.manual_seed(12345)
        self.atom_encoder = AtomEncoder(emb_dim = node_emb_dim) # Pytorch Module class w/ learnable parameters
        self.bond_encoder = BondEncoder(emb_dim = edge_emb_dim) # Pytorch Module class w/ learnable parameters

        self.conv1 = TransformerConv(node_emb_dim, hidden_channels, **tconv_params)
#GCNConv(node_emb_dim, hidden_channels)
        self.conv2 = TransformerConv(hidden_channels, hidden_channels, **tconv_params)#GCNConv(hidden_channels, hidden_channels)
        self.conv3 = TransformerConv(hidden_channels, hidden_channels, **tconv_params)#GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.atom_encoder(x)
        edge_emb = self.bond_encoder(edge_attr)
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index, edge_emb)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_emb)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_emb)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x