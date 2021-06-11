import torch
import torch.nn as nn


class PreNormResidual(nn.Module):
    """
    Encapsulates the logic of performing layer normalization and applying 
    a residual connection.

    How it works
    ------------
    The forward pass takes in a tensor of shape (B x * x M) where M is the 
    equal to `dim`. Then the module applies the specified normalization (`norm`),
    then propagates the normalized output through the target layer (`fn`). 
    
    The output from the target layer is then added to the input.
    """

    def __init__(self, fn: nn.Module, dim: int, norm: nn.Module = nn.LayerNorm) -> None:
        """
        Arguments
        ---------
        fn : nn.Module
            module to be wrapped
        dim : int 
            size of the embedding dimension
        norm : nn.Module
            normalization layer 
        """
        super(PreNormResidual, self).__init__()
        self.fn = fn
        self.norm = norm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B x * x M -> B x * x M
        return self.fn(self.norm(x)) + x
