import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """
    A stack of `Linear` layers with `GELU` activation and `Dropout`.

    How it works
    ------------
    The forward pass takes in an input tensor of shape B x L x M where 
    M is equal to `dim`. The input is passed through a linear layer to 
    get an output of shape B x L x N where N is the `hidden_dim`. After
    this a `GELU` activation is applied. Next the output of the linear 
    layer goes through a dropout layer. After this the output tensor of
    the dropout layer goes into a linear layer and comes out with the 
    shape B x L x M. The B x L x M tensor is then passed through a final
    dropout layer.

    References
    ----------
    1. https://github.com/lucidrains/vit-pytorch
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        """
        Arguments
        ---------
        dim : int 
            dimension of the input linear layer as well as the output layer.
        hidden_dim : int 
            hidden dimension of the linear layers. 
        dropout : float
            dropout rate of the dropout layer.
        """
        super(FeedForward, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B x L x M -> B x L x M
        return self.model(x)
