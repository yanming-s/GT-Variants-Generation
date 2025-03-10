import torch
import torch.nn as nn
from models.layers.attention_dense import *
from models.layers.attention_spares import *


class Attention_Head_GTv1(nn.Module):
    """
    Attention Head of GTv1
    """
    def __init__(self, dense_attention: bool, hidden_dim: int, head_dim: int, dropout: float=0.0):
        super().__init__()
        # Dense Implementation of Attention Mechanism
        if dense_attention:
            self.attention = attention_vanilla_dense(hidden_dim, head_dim, dropout)
        # Sparse Implementation of Attention Mechanism
        else:
            # TODO: Implement sparse attention
            pass
    def forward(self, x, e, node_mask):
        return self.attention(x, e, node_mask)


class Attention_Head_GTv2(nn.Module):
    """
    Attention Head of GTv2
    """
    def __init__(self, dense_attention: bool, integration: str, hidden_dim: int, head_dim: int, dropout: float=0.0):
        super().__init__()
        # Dense Implementation of Attention Mechanism
        if dense_attention:
            self.self_att = attention_node_to_node_dense(hidden_dim, head_dim, dropout)
            self.cross_att_node = attention_edge_to_node_dense(hidden_dim, head_dim, dropout)
            self.cross_att_edge = attention_node_to_edge_dense(hidden_dim, head_dim, dropout)
            self.integration = integration
            if integration == "weighted":
                self.alpha = 0.5
            elif integration == "gated":
                self.gated_mlp = nn.Sequential(
                    nn.Linear(2 * head_dim, head_dim),
                    nn.Sigmoid()
                )
            elif integration == "mixed":
                self.mixed_mlp = nn.Linear(2 * head_dim, head_dim)
            else:
                self.weight_mlp_1 = nn.Linear(head_dim, head_dim)
                self.weight_mlp_2 = nn.Linear(head_dim, head_dim)
        # Sparse Implementation of Attention
        else:
            # TODO: Implement sparse attention
            pass
    def forward(self, x, e, node_mask):
        x_cross, _ = self.cross_att_node(x, e, node_mask)
        _, e = self.cross_att_edge(x, e, node_mask)
        x_self, _ = self.self_att(x, e, node_mask)
        if self.integration == "weighted":
            x = self.alpha * x_cross + (1 - self.alpha) * x_self
        elif self.integration == "gated":
            g = self.gated_mlp(torch.cat([x_cross, x_self], dim=-1))
            x = g * x_cross + (1 - g) * x_self
        elif self.integration == "mixed":
            x = self.mixed_mlp(torch.cat([x_cross, x_self], dim=-1))
        else:
            x = self.weight_mlp_1(x_self) + self.weight_mlp_2(x_self) * x_cross + x_cross
        return x, e



class Attention_Head_GTv3(nn.Module):
    """
    Attention Head of GTv3
    """
    def __init__(self, dense_attention: bool, layers: list[dict], hidden_dim: int, head_dim: int, dropout: float=0.0):
        super().__init__()
        self.used_node_feature_idx = []
        self.used_edge_feature_idx = []
        layer_list = []
        # Dense Implementation of Attention Mechanism
        if dense_attention:
            for layer in layers:
                shrink_node_dim = layer["shrink_node_dim"]
                shrink_edge_dim = layer["shrink_edge_dim"]
                self.used_node_feature_idx.append(layer["used_node_feature_idx"])
                self.used_edge_feature_idx.append(layer["used_edge_feature_idx"])
                if layer["attention_type"] == "cross_attention_edge_to_node":
                    layer_list.append(
                        attention_edge_to_node_dense(hidden_dim, head_dim, dropout, shrink_node_dim, shrink_edge_dim)
                    )
                elif layer["attention_type"] == "cross_attention_node_to_edge":
                    layer_list.append(
                        attention_node_to_edge_dense(hidden_dim, head_dim, dropout, shrink_node_dim, shrink_edge_dim)
                    )
                elif layer["attention_type"] == "self_attention_node_to_node":
                    layer_list.append(attention_node_to_node_dense(hidden_dim, head_dim, dropout, shrink_node_dim))
                else:
                    raise NotImplementedError(f"Attention type {layer['attention_type']} not supported.")
        # Sparse Implementation of Attention Mechanism
        else:
            # TODO: Implement sparse attention
            pass
        self.layers = nn.ModuleList(layer_list)
    def forward(self, x, e, node_mask):
        node_feature_list = [x]
        edge_feature_list = [e]
        for idx, layer in enumerate(self.layers):
            x_layer, e_layer = layer(
                node_feature_list[self.used_node_feature_idx[idx]],
                edge_feature_list[self.used_edge_feature_idx[idx]],
                node_mask
            )
            node_feature_list.append(x_layer)
            edge_feature_list.append(e_layer)
        # Keep only the final results and free intermediate GPU tensors.
        final_node = node_feature_list[-1]
        final_edge = edge_feature_list[-1]
        del node_feature_list[:-1]
        del edge_feature_list[:-1]
        torch.cuda.empty_cache()
        return final_node, final_edge


class MHA(nn.Module):
    """
    Multi-head Attention Layer
    """
    def __init__(self, layer_type: dict[str], dense_attention: bool, hidden_dim: int, num_heads: int, dropout: float=0.0):  
        super().__init__()
        head_dim = hidden_dim // num_heads
        self.WOx = nn.Linear(hidden_dim, hidden_dim)
        self.WOe = nn.Linear(hidden_dim, hidden_dim)
        self.drop_x = nn.Dropout(dropout)
        self.drop_e = nn.Dropout(dropout)
        # Attention Head of GTv1
        if layer_type["name"] == "gtv1":
            self.heads = nn.ModuleList([Attention_Head_GTv1(dense_attention, hidden_dim, head_dim, dropout) for _ in range(num_heads)])
        # Attention Head of GTv2
        elif layer_type["name"] == "gtv2":
            integration = layer_type["integration"]
            # Judge the integration method
            if integration not in ["weighted", "gated", "mixed", "film"]:
                raise NotImplementedError(f"Integration method {integration} not supported.")
            self.heads = nn.ModuleList([Attention_Head_GTv2(dense_attention, integration, hidden_dim, head_dim, dropout) for _ in range(num_heads)])
        # Attention Head of GTv3
        elif layer_type["name"] == "gtv3":
            layers = layer_type["layers"]
            self.heads = nn.ModuleList([Attention_Head_GTv3(dense_attention, layers, hidden_dim, head_dim, dropout) for _ in range(num_heads)])
        else:
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
