import torch


class DataBuffer:
    def __init__(self, x: torch.Tensor, e: torch.Tensor):
        self.x = x
        self.e = e

    def type_as(self, x: torch.Tensor):
        self.x = self.x.type_as(x).to(x.device)
        self.e = self.e.type_as(x).to(x.device)
        return self
    
    def mask(self, node_mask: torch.Tensor):
        x_mask = node_mask.unsqueeze(-1) # [bs, n, 1]
        e_mask1 = x_mask.unsqueeze(2)    # [bs, n, 1, 1]
        e_mask2 = x_mask.unsqueeze(1)    # [bs, 1, n, 1]
        self.x = self.x * x_mask
        self.e = self.e * e_mask1 * e_mask2
        assert torch.allclose(self.e, torch.transpose(self.e, 1, 2))
        return self