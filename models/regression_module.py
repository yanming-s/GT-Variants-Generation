import torch
import torch.nn as nn
import lightning as L

from models.utils import to_dense
from models.layers.transformer import Graph_Transformer_Layer


class Graph_Transformer(nn):
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
            Graph_Transformer_Layer(
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
    def __init__(self, cfg, logger, datasetInfo):
        super().__init__()
        # General Settings
        self.cfg = cfg
        self.logger = logger
        self.model = Graph_Transformer(cfg, datasetInfo)
        # Training Settings
        self.loss = nn.MSELoss()
        self.running_loss = 0.0
        self.batch_cnt = 0

    '''
    Training Steps
    '''
    def forward(self, x, e, node_mask):
        return self.model(x, e, node_mask)

    def training_step(self, batch_data, batch_idx):
        data, node_mask = to_dense(batch_data.x, batch_data.edge_index, batch_data.edge_attr, batch_data.batch)
        data = data.mask(node_mask)
        x, e = data.x, data.e
        pred = self.forward(x, e, node_mask)
        loss = self.loss(pred, batch_data.y)
        return loss
    
    def on_train_epoch_start(self):
        self.logger.info(f"Epoch {self.current_epoch}")
        self.running_loss = 0.0
        self.batch_cnt = 0

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        self.running_loss += outputs.detach().item()
        self.batch_cnt += 1

    def on_train_epoch_end(self, outputs):
        self.logger.info(f"Training Loss: {self.running_loss / self.batch_cnt}")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.cfg.run_config.lr,
                                 weight_decay=self.cfg.run_config.weight_decay)
    
    '''
    Validation Steps
    '''
