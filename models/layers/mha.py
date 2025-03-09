import torch
import torch.nn as nn
from models.layers.attention_dense import *
from models.layers.attention_spares import *


class Attention_Head_v1(nn.Module):
    def __init__(self, dense_attention: bool, hidden_dim: int, head_dim: int, dropout: float=0.0):
        super().__init__()
        if dense_attention:
            self.attention = attention_vanilla_dense(hidden_dim, head_dim, dropout)
        else:
            # TODO: Implement sparse attention
            raise NotImplementedError("Only dense attention is supported")
    def forward(self, x, e, node_mask):
        return self.attention(x, e, node_mask)


class Attention_Head_v2(nn.Module):
    def __init__(self, dense_attention: bool, integration: str, hidden_dim: int, head_dim: int, dropout: float=0.0):
        super().__init__()
        self.inte_layer = None
        if dense_attention:
            self.self_att = attention_node_to_node_dense(hidden_dim, head_dim, dropout)
            self.cross_att_node = attention_edge_to_node_dense(hidden_dim, head_dim, dropout)
            self.cross_att_edge = attention_node_to_edge_dense(hidden_dim, head_dim, dropout)
            # integration layer
            if integration == "add":
                self.inte_layer = nn.Linear(hidden_dim, hidden_dim)
            elif integration == "concat":
                self.inte_layer = nn.Linear(hidden_dim*3, hidden_dim)
            elif integration == "none":
                pass
            else:
                raise NotImplementedError(f"Integration method {integration} not supported.")
        else:
            # TODO: Implement sparse attention
            raise NotImplementedError("Only dense attention is supported")
    def forward(self, x, e, node_mask):
        pass


class MHA(nn.Module):
    def __init__(self, layer_type: dict[str], dense_attention: bool, hidden_dim: int, num_heads: int, dropout: float=0.0):  
        super().__init__()
        head_dim = hidden_dim // num_heads
        self.WOx = nn.Linear(hidden_dim, hidden_dim)
        self.WOe = nn.Linear(hidden_dim, hidden_dim)
        self.drop_x = nn.Dropout(dropout)
        self.drop_e = nn.Dropout(dropout)
        if layer_type["name"] == "GTv1":
            self.heads = nn.ModuleList([Attention_Head_v1(dense_attention, hidden_dim, head_dim, dropout) for _ in range(num_heads)])
        else:
            # TODO: Implement other layer types
            raise NotImplementedError(f"Layer type {layer_type} not supported.")
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
