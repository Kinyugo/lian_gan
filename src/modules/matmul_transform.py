import torch
import torch.nn as nn


class MatMulTransform(nn.Module):
    """
    Implements matrix multiplication transformation as token mixing technique 
    instead of self-attention within a transformer.

    How it works
    ------------
    During the forward pass the input tensor of shape B x L x D are multiplied with
    their transpose of the last two dimensions to obtain a tensor of shape 
    B x L x D. The mean of this tensor is then taken along the last dimension (D)
    and scaled by the size of the last dimension (D) obtaining a tensor of shape
    B x L x 1. The B x L x 1 tensor is then multiplied with its transpose to obtain
    another B x L x L tensor. The mean of this tensor is then taken along the last dimension (D)
    and scaled by the size of the second dimension (L).

    The final tensor is then element-wise multiplied with the input tensor to obtain a tensor of shape 
    B x L x D. The B x L x D tensor is then added with a learnable parameter alpha (α).

    References
    ----------
    1. https://github.com/Kinyugo/blind_net
    """

    def __init__(self, l_dim: int, dropout: float) -> None:
        """
        Arguments
        ---------
        l_dim : int 
            height of the embedding, this is usually the number of patches
        dropout : float
            dropout probability 
        """
        super(MatMulTransform, self).__init__()
        self.alpha_parameter = nn.Parameter(torch.ones(l_dim, 1))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B x L x D @ B x D x L -> B x L x 1
        y = (x @ x.transpose(2, 1)).mean(dim=-1, keepdim=True) / x.size(-1)
        # B x L x 1 @ B x 1 x L -> B x L x 1
        y = (y @ y.transpose(2, 1)).mean(dim=-1, keepdim=True) / x.size(1)
        # B x L x 1 @ B x L x D -> B x L x D
        y = y * x
        y = self.alpha_parameter + y

        return self.dropout(y)
