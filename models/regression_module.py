import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
import time

from models.utils import to_dense
from models.layers.transformer import Graph_Transformer_Layer


class Graph_Transformer(nn.Module):
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
        e = (e + e.permute(0, 2, 1, 3)) / 2
        e = self.drop_in_e(e)
        e = e * edge_mask_1 * edge_mask_2
        # Transformer Layers
        for layer in self.transformer_layers:
            x, e = layer(x, e, mask_origin)
        # Output Layers
        graph_token = x.mean(dim=1) # [bs, d]
        graph_token = self.mlp_out_x(self.norm_out_x(graph_token)) # [bs, 1]
        return graph_token


class Regression_Dense_Module(LightningModule):
    def __init__(self, cfg, logger, datasetInfo):
        super().__init__()
        # General Settings
        self.cfg = cfg
        self.custom_logger = logger
        self.model = Graph_Transformer(cfg, datasetInfo)
        self.pred_target = cfg.run_config.pred_target
        # Training Settings
        self.loss = nn.MSELoss()
        self.running_loss = 0.0
        self.batch_cnt = 0
        self.epoch_time = 0.0
        # Lightning Module Settings
        self.save_hyperparameters(ignore="logger")

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
        # Compute loss based on the target
        if self.pred_target == "logp":
            loss = self.loss(pred, batch_data.logp)
        elif self.pred_target == "mwt":
            loss = self.loss(pred, batch_data.mwt)
        else:
            raise ValueError(f"Target {self.pred_target} not supported.")
        self.running_loss += loss.detach().item()
        self.batch_cnt += 1
        return loss

    def on_train_epoch_start(self):
        self.running_loss = 0.0
        self.batch_cnt = 0
        self.epoch_time = time.time()
    
    def on_train_epoch_end(self):
        train_loss = self.running_loss / self.batch_cnt
        epoch_time = (time.time() - self.epoch_time) / 60
        self.custom_logger.info(f"Epoch {self.current_epoch + 1} - training loss:{train_loss:.6f}  time:{epoch_time:.2f} min")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.cfg.run_config.lr,
                                 weight_decay=self.cfg.run_config.weight_decay)
    
    '''
    Validation Steps
    '''
    def validation_step(self, batch_data, batch_idx):
        data, node_mask = to_dense(batch_data.x, batch_data.edge_index, batch_data.edge_attr, batch_data.batch)
        data = data.mask(node_mask)
        x, e = data.x, data.e
        pred = self.forward(x, e, node_mask)
        # Compute loss based on the target
        if self.pred_target == "logp":
            loss = self.loss(pred, batch_data.logp)
        elif self.pred_target == "mwt":
            loss = self.loss(pred, batch_data.mwt)
        else:
            raise ValueError(f"Target {self.pred_target} not supported.")
        self.running_loss += loss.detach().item()
        self.batch_cnt += 1
        return loss
    
    def on_validation_epoch_start(self):
        self.running_loss = 0.0
        self.batch_cnt = 0

    def on_validation_epoch_end(self):
        val_loss = self.running_loss / self.batch_cnt
        self.custom_logger.info(f"Epoch {self.current_epoch + 1} - validation loss: {val_loss:.6f}")
        self.log("val_loss", val_loss)

    '''
    Test Steps
    '''
    def test_step(self, batch_data, batch_idx):
        data, node_mask = to_dense(batch_data.x, batch_data.edge_index, batch_data.edge_attr, batch_data.batch)
        data = data.mask(node_mask)
        x, e = data.x, data.e
        pred = self.forward(x, e, node_mask)
        # Compute loss based on the target
        if self.pred_target == "logp":
            loss = self.loss(pred, batch_data.logp)
        elif self.pred_target == "mwt":
            loss = self.loss(pred, batch_data.mwt)
        else:
            raise ValueError(f"Target {self.pred_target} not supported.")
        self.running_loss += loss.detach().item()
        self.batch_cnt += 1
        return loss
    
    def on_test_start(self):
        self.custom_logger.info("Start Testing...")
    
    def on_test_epoch_start(self):
        self.running_loss = 0.0
        self.batch_cnt = 0

    def on_test_epoch_end(self):
        self.custom_logger.info(f"Test Loss: {self.running_loss / self.batch_cnt:.6f}")
        self.log("test_loss", self.running_loss / self.batch_cnt)
