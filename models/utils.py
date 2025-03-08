import torch
from torch_geometric.utils import to_dense_batch, to_dense_adj, remove_self_loops


class DataBuffer:
    """
    A data buffer that stores and masks node and edge data.
    """
    def __init__(self, x: torch.Tensor, e: torch.Tensor):
        self.x = x
        self.e = e

    def type_as(self, x: torch.Tensor):
        """
        Move the data buffer to the same device and data type as the input tensor.
        - x: torch.Tensor, the input tensor.
        - return: DataBuffer, the data buffer.
        """
        self.x = self.x.type_as(x).to(x.device)
        self.e = self.e.type_as(x).to(x.device)
        return self
    
    def mask(self, node_mask: torch.Tensor):
        """
        Mask the data buffer using the node mask.
        - node_mask: torch.Tensor, the node mask.
        - return: DataBuffer, the data buffer.
        """
        x_mask = node_mask.unsqueeze(-1) # [bs, n, 1]
        e_mask1 = x_mask.unsqueeze(2)    # [bs, n, 1, 1]
        e_mask2 = x_mask.unsqueeze(1)    # [bs, 1, n, 1]
        self.x = self.x * x_mask
        self.e = self.e * e_mask1 * e_mask2
        assert torch.allclose(self.e, torch.transpose(self.e, 1, 2))
        return self
    

def to_dense(x, edge_index, edge_attr, batch):
    """
    Convert the input data to dense format.
    - x: torch.Tensor, the node features.
    - edge_index: torch.Tensor, the edge index.
    - edge_attr: torch.Tensor, the edge attributes.
    - batch: torch.Tensor, the batch index.
    - return: DataBuffer, the data buffer.
    """
    x, node_mask = to_dense_batch(x=x, batch=batch)
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    max_num_nodes = x.size(1)
    e = to_dense_adj(edge_index=edge_index, batch=batch, edge_attr=edge_attr, max_num_nodes=max_num_nodes)
    e = encode_no_edge(e)
    return DataBuffer(x=x, e=e), node_mask


def encode_no_edge(e):
    """
    Encode the no edge information in the edge tensor.
    - e: torch.Tensor, the edge tensor.
    - return: torch.Tensor, the edge tensor.
    """
    assert len(e.shape) == 4
    if e.shape[-1] == 0:
        return e
    no_edge = torch.sum(e, dim=3) == 0
    first_elt = e[:, :, :, 0]
    first_elt[no_edge] = 1
    e[:, :, :, 0] = first_elt
    diag = torch.eye(e.shape[1], dtype=torch.bool).unsqueeze(0).expand(e.shape[0], -1, -1)
    e[diag] = 0
    return e
