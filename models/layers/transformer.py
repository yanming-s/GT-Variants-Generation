import torch
import torch.nn as nn

from models.layers.mha import MHA


class Graph_Transformer_Layer(nn.Module):
    """
    Graph Transformer Layer
    """
    def __init__(
            self, layer_type: dict[str], dense_attention: bool, hidden_dim: int,
            mlp_dim: int, num_heads: int, dropout: float=0.0
    ):
        super().__init__()
        # Auxiliary Layers
        self.norm_in_x = nn.LayerNorm(hidden_dim)
        self.norm_in_e = nn.LayerNorm(hidden_dim)
        self.norm_out_x = nn.LayerNorm(hidden_dim)
        self.norm_out_e = nn.LayerNorm(hidden_dim)
        self.mlp_x = nn.Sequential(nn.Linear(hidden_dim, mlp_dim), nn.LeakyReLU(), nn.Linear(mlp_dim, hidden_dim))
        self.mlp_e = nn.Sequential(nn.Linear(hidden_dim, mlp_dim), nn.LeakyReLU(), nn.Linear(mlp_dim, hidden_dim))
        self.drop_out_x = nn.Dropout(dropout)
        self.drop_out_e = nn.Dropout(dropout)
        # Multi-Head Attention Layer
        self.MHA = MHA(layer_type, dense_attention, hidden_dim, num_heads, dropout)
    
    def forward(self, x: torch.Tensor, e: torch.Tensor, node_mask: torch.Tensor):
        # Masking
        x_mask = node_mask.unsqueeze(-1) # [bs, n, 1]
        e_mask_1 = x_mask.unsqueeze(2)   # [bs, n, 1, 1]
        e_mask_2 = x_mask.unsqueeze(1)   # [bs, 1, n, 1]
        # Forward Pass
        x = self.norm_in_x(x) # [bs, n, d]
        e = self.norm_in_e(e) # [bs, n, n, d]
        x_MHA, e_MHA = self.MHA(x, e, node_mask) # [bs, n, d], [bs, n, n, d]
        x = x + x_MHA                   # [bs, n, d]
        x_mlp = self.mlp_x(self.norm_out_x(x)) # [bs, n, d]
        x_mlp = x_mlp * x_mask          # [bs, n, d]
        x = x + x_mlp                   # [bs, n, d]
        x = self.drop_out_x(x)          # [bs, n, d]
        e = e + e_MHA                       # [bs, n, n, d]
        e_mlp = self.mlp_e(self.norm_out_e(e))     # [bs, n, n, d]
        e_mlp = e_mlp * e_mask_1 * e_mask_2 # [bs, n, n, d]
        e = e + e_mlp                       # [bs, n, n, d]
        e = self.drop_out_e(e)              # [bs, n, n, d]
        return x, e                         # [bs, n, d], [bs, n, n, d]
