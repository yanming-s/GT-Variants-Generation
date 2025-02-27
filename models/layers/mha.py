import torch
import torch.nn as nn
from models.layers.attention_dense import *
from models.layers.attention_spares import *


class MHA_v1(nn.Module):
    def __init__(self, dense_attention: bool, hidden_dim: int, num_heads: int, dropout: float=0.0):  
        super().__init__()
        head_dim = hidden_dim // num_heads
        self.WOx = nn.Linear(hidden_dim, hidden_dim)
        self.WOe = nn.Linear(hidden_dim, hidden_dim)
        self.drop_x = nn.Dropout(dropout)
        self.drop_e = nn.Dropout(dropout)
        if dense_attention:
            self.heads = nn.ModuleList([attention_v1(hidden_dim, head_dim, dropout) for _ in range(num_heads)])
        else:
            # TODO: Implement sparse attention
            raise NotImplementedError("Only dense attention is supported")

    def forward(self, x, e, node_mask):
        x_mask = node_mask.unsqueeze(-1) # [bs, n, 1]
        e_mask_1 = x_mask.unsqueeze(2)  # [bs, n, 1, 1]
        e_mask_2 = x_mask.unsqueeze(1)  # [bs, 1, n, 1]
        x_MHA = []
        e_MHA = []    
        for head in self.heads:
            x_HA, e_HA = head(x,e, node_mask) # [bs, n, d_head], [bs, n, n, d_head]
            x_MHA.append(x_HA)
            e_MHA.append(e_HA)
        x = self.WOx(torch.cat(x_MHA, dim=2)) # [bs, n, d]
        x = x * x_mask                        # [bs, n, d]
        x = self.drop_x(x)                    # [bs, n, d]
        e = self.WOe(torch.cat(e_MHA, dim=3)) # [bs, n, n, d]
        e = e * e_mask_1 * e_mask_2           # [bs, n, n, d]
        e = self.drop_e(e)                    # [bs, n, n, d]
        return x, e                           # [bs, n, d], [bs, n, n, d]
