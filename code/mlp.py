import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(torch.nn.Module):
    """Two layered perceptron."""

    def __init__(self, indim, hiddim, outdim, dropout=False, dropoutp=0.1, output_each_layer=False):
        """Initialize two-layered perceptron.

        Args:
            indim (int): Input dimension
            hiddim (int): Hidden layer dimension
            outdim (int): Output layer dimension
            dropout (bool, optional): Whether to apply dropout or not. Defaults to False.
            dropoutp (float, optional): Dropout probability. Defaults to 0.1.
            output_each_layer (bool, optional): Whether to return outputs of each layer as a list. Defaults to False.
        """
        super(MLP, self).__init__()
        self.fc = nn.Linear(indim, hiddim)
        self.fc2 = nn.Linear(hiddim, outdim)
        self.dropout_layer = torch.nn.Dropout(dropoutp)
        self.dropout = dropout
        self.output_each_layer = output_each_layer
        self.lklu = nn.LeakyReLU(0.2)

    def forward(self, x):
        """Apply MLP to Input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        output = F.relu(self.fc(x))
        if self.dropout:
            output = self.dropout_layer(output)
        output2 = self.fc2(output)
        if self.dropout:
            output2 = self.dropout_layer(output)
        if self.output_each_layer:
            return [0, x, output, self.lklu(output2)]
        return output2
