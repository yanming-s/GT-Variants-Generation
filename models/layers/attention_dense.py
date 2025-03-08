import torch
import torch.nn as nn


class attention_v1_dense(nn.Module):
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
