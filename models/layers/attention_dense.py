import torch
import torch.nn as nn


class attention_vanilla_dense(nn.Module):
    def __init__(self, hidden_dim, head_dim, drop=0.0):
        super().__init__()
        self.Q = nn.Linear(hidden_dim, head_dim)
        self.K = nn.Linear(hidden_dim, head_dim)
        self.E = nn.Linear(hidden_dim, head_dim)
        self.V = nn.Linear(hidden_dim, head_dim)
        self.sqrt_d = torch.sqrt(torch.tensor(head_dim))
        self.drop_att = nn.Dropout(drop)
        self.Ni = nn.Linear(hidden_dim, head_dim)
        self.Nj = nn.Linear(hidden_dim, head_dim)
    def forward(self, x, e, node_mask):
        x_mask = node_mask.unsqueeze(-1) # [bs, n, 1]
        e_mask_1 = x_mask.unsqueeze(2)   # [bs, n, 1, 1]
        e_mask_2 = x_mask.unsqueeze(1)  # [bs, 1, n, 1]
        Q = self.Q(x) * x_mask # [bs, n, d_head]
        K = self.K(x) * x_mask # [bs, n, d_head]
        V = self.V(x) * x_mask # [bs, n, d_head]
        Q = Q.unsqueeze(2) # [bs, n, 1, d_head]
        K = K.unsqueeze(1) # [bs, 1, n, d_head]
        E = self.E(e) * e_mask_1 * e_mask_2 # [bs, n, n, d_head]
        Ni = self.Ni(x).unsqueeze(2) * e_mask_1 # [bs, n, 1, d_head]
        Nj = self.Nj(x).unsqueeze(1) * e_mask_2 # [bs, 1, n, d_head]
        e = (Ni + Nj + E) * e_mask_1 * e_mask_2 # [bs, n, n, d_head]
        Att = (Q * e * K).sum(dim=3) / self.sqrt_d # [bs, n, n]
        att_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        att_mask = att_mask.bool()
        Att = Att.masked_fill(~att_mask, -1e9)
        Att = torch.softmax(Att, dim=1)            # [bs, n, n]
        Att = self.drop_att(Att)                   # [bs, n, n]
        x = (Att @ V) * x_mask                     # [bs, n, d_head]
        return x, e                  # [bs, n, d_head], [bs, n, n, d_head]


class attention_node_to_node_dense(nn.Module):
    def __init__(self, hidden_dim, head_dim, drop=0.0, shrink_node_dim=False):
        super().__init__()
        if shrink_node_dim:
            # Shrink the node dimension for GTv3
            hidden_dim = head_dim
        self.Q = nn.Linear(hidden_dim, head_dim)  # For node queries
        self.K = nn.Linear(hidden_dim, head_dim)  # For node keys
        self.V = nn.Linear(hidden_dim, head_dim)  # For node values
        self.sqrt_d = torch.sqrt(torch.tensor(head_dim))
        self.dropout = nn.Dropout(drop)
    def forward(self, x, e, node_mask):
        x_mask = node_mask.unsqueeze(-1)  # [bs, n, 1]
        Q = self.Q(x) * x_mask  # [bs, n, d_head]
        K = self.K(x) * x_mask  # [bs, n, d_head]
        V = self.V(x) * x_mask  # [bs, n, d_head]
        Q = Q.unsqueeze(2)  # [bs, n, 1, d_head]
        K = K.unsqueeze(1)  # [bs, 1, n, d_head]
        Att = (Q * K).sum(dim=-1) / self.sqrt_d  # [bs, n, n]
        att_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)  # [bs, n, n]
        att_mask = att_mask.bool()
        Att = Att.masked_fill(~att_mask, -1e9)
        Att = torch.softmax(Att, dim=-1)  # [bs, n, n]
        Att = self.dropout(Att)
        x = Att @ V * x_mask  # [bs, n, d_head]
        return x, e


class attention_edge_to_node_dense(nn.Module):
    def __init__(self, hidden_dim, head_dim, drop=0.0, shrink_node_dim=False, shrink_edge_dim=False):
        super().__init__()
        # Shrink the dimension for GTv3
        node_dim = head_dim if shrink_node_dim else hidden_dim
        edge_dim = head_dim if shrink_edge_dim else hidden_dim
        self.Q_node = nn.Linear(node_dim, head_dim)
        self.K_edge = nn.Linear(edge_dim, head_dim)
        self.V_edge = nn.Linear(edge_dim, head_dim)
        self.sqrt_d = torch.sqrt(torch.tensor(head_dim))
        self.drop_att = nn.Dropout(drop)
    def forward(self, x, e, node_mask):
        x_mask = node_mask.unsqueeze(-1) # [bs, n, 1]
        e_mask_1 = x_mask.unsqueeze(2)   # [bs, n, 1, 1]
        e_mask_2 = x_mask.unsqueeze(1)   # [bs, 1, n, 1]
        Q_node = self.Q_node(x) * x_mask # [bs, n, d_head]
        K_edge = self.K_edge(e) * e_mask_1 * e_mask_2 # [bs, n, n, d_head]
        V_edge = self.V_edge(e) * e_mask_1 * e_mask_2 # [bs, n, n, d_head]
        Q_node = Q_node.unsqueeze(2) # [bs, n, 1, d_head]
        Att = (Q_node * K_edge).sum(dim=3) / self.sqrt_d # [bs, n, n]
        att_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2) # [bs, n, n]
        att_mask = att_mask.bool()
        Att = Att.masked_fill(~att_mask, -1e9)
        Att = torch.softmax(Att, dim=2)
        Att = self.drop_att(Att)
        Att = Att.unsqueeze(-1) # [bs, n, n, 1]
        x = (Att * V_edge) * e_mask_1 * e_mask_2 # [bs, n, n, d_head]
        x = x.sum(dim=2) # [bs, n, d_head]
        return x, e # [bs, n, d_head]


class attention_node_to_edge_dense(nn.Module):
    def __init__(self, hidden_dim, head_dim, drop=0.0, shrink_node_dim=False, shrink_edge_dim=False):
        super().__init__()
        # Shrink the dimension for GTv3
        node_dim = head_dim if shrink_node_dim else hidden_dim
        edge_dim = head_dim if shrink_edge_dim else hidden_dim
        self.Q_edge = nn.Linear(edge_dim, head_dim)
        self.K_node = nn.Linear(node_dim, head_dim)
        self.V_node = nn.Linear(node_dim, head_dim)
        self.sqrt_d = torch.sqrt(torch.tensor(head_dim))
        self.drop_att = nn.Dropout(drop)
    def forward(self, x, e, node_mask):
        x_mask = node_mask.unsqueeze(-1) # [bs, n, 1]
        e_mask_1 = x_mask.unsqueeze(2)   # [bs, n, 1, 1]
        e_mask_2 = x_mask.unsqueeze(1)   # [bs, 1, n, 1]
        Q_edge = self.Q_edge(e) * e_mask_1 * e_mask_2 # [bs, n, n, d_head]
        K_node = self.K_node(x) * x_mask # [bs, n, d_head]
        V_node = self.V_node(x) * x_mask # [bs, n, d_head]
        K_i = K_node.unsqueeze(1).expand(-1, e.size(1), -1, -1) # [bs, n, n, d_head]
        V_i = V_node.unsqueeze(1).expand(-1, e.size(1), -1, -1) # [bs, n, n, d_head]
        K_j = K_node.unsqueeze(2).expand(-1, -1, e.size(1), -1) # [bs, n, n, d_head]
        V_j = V_node.unsqueeze(2).expand(-1, -1, e.size(1), -1) # [bs, n, n, d_head]
        att_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2) # [bs, n, n]
        att_mask = att_mask.bool()
        Att_i = torch.exp((Q_edge * K_i).sum(dim=-1) / self.sqrt_d) # [bs, n, n]
        Att_i = Att_i.masked_fill(~att_mask, 1e-9)
        Att_j = torch.exp((Q_edge * K_j).sum(dim=-1) / self.sqrt_d) # [bs, n, n]
        Att_j = Att_j.masked_fill(~att_mask, 1e-9)
        Att_sum = Att_i + Att_j
        Att_sum = Att_sum.masked_fill(~att_mask, 1e-9)
        Att_i = self.drop_att(Att_i / Att_sum)
        Att_j = self.drop_att(Att_j / Att_sum)
        e = Att_i.unsqueeze(-1) * V_i + Att_j.unsqueeze(-1) * V_j # [bs, n, n, d_head]
        e = e * e_mask_1 * e_mask_2 # [bs, n, n, d_head]
        return x, e
