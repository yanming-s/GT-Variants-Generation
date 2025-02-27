import torch
import torch.nn as nn
import lightning as L

from models.utils import DataBuffer
from models.layers.transformer import Transformer_Layer


class Transformer_Model(nn):
    def __init__(self, cfg, datasetInfo):
        super().__init__()
        # Model Settings
        self.hidden_dim = cfg.model.hidden_dim
        self.num_attention_layers = cfg.model.num_attention_layers
        self.num_attention_heads = cfg.model.num_attention_heads
        self.dropout = cfg.model.dropout
        # Transformer Layer Settings
        self.layer_type = cfg.transformer_layer.name
        self.dense_attention = cfg.transformer_layer.dense_attention
        self.mlp_dim = cfg.transformer_layer.mlp_dim
        # Input and Output Layers
        self.mlp_in_x = nn.Linear(datasetInfo.num_node_type, self.hidden_dim)
        self.mlp_in_e = nn.Linear(datasetInfo.num_edge_type, self.hidden_dim)
        self.drop_in_x = nn.Dropout(self.dropout)
        self.drop_in_e = nn.Dropout(self.dropout)
        self.norm_out_x = nn.LayerNorm(self.hidden_dim)
        self.mlp_out_x = nn.Linear(self.hidden_dim, 1)
        # Transformer Layers
        self.transformer_layers = nn.ModuleList([
            Transformer_Layer(
                self.layer_type, self.dense_attention, self.hidden_dim,
                self.mlp_dim, self.num_attention_heads, self.dropout
            ) for _ in range(self.num_attention_layers)
        ])
    def forward(self, x: torch.Tensor, e: torch.Tensor, node_mask: torch.Tensor):
        # Masking
        mask_origin = node_mask
        node_mask = node_mask.unsqueeze(-1) # [bs, n, 1]
        edge_mask_1 = node_mask.unsqueeze(2) # [bs, n, 1, 1]
        edge_mask_2 = node_mask.unsqueeze(1) # [bs, 1, n, 1]
        # Input Layers
        x = self.drop_in_x(self.mlp_in_x(x))
        x = x * node_mask
        e = self.mlp_in_e(e)
        e = (e + e.permute(0, 2, 1)) / 2
        e = self.drop_in_e(e)
        e = e * edge_mask_1 * edge_mask_2
        # Transformer Layers
        for layer in self.transformer_layers:
            x, e = layer(x, e, mask_origin)
        # Output Layers
        graph_token = x.mean(dim=1) # [bs, d]
        graph_token = self.mlp_out_x(self.norm_out_x(graph_token)) # [bs, 1]
        return graph_token


class Regression_Module(L.LightningModule):
    def __init__(self, cfg, datasetInfo):
        super().__init__()
        self.model = Transformer_Model(cfg, datasetInfo)
        
    def forward(self, data):
        pass
