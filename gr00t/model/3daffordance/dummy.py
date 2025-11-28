import torch

import torch.nn as nn


class DummyMLP(nn.Module):
    """
    Dummy MLP replacement that forwards a random tensor through Linear+ReLU layers.

    Args:
        no_grad (bool): if True, parameters will not require grads and forward runs under torch.no_grad().
        output_size (int): output feature size.
        input_size (int): input feature size for the random tensor.
        layer_size (int): number of hidden Linear+ReLU layers before the final Linear.
    """

    def __init__(self, no_grad=True, output_size=1024, input_size=512, layer_size=8):
        super().__init__()
        self.no_grad_mode = bool(no_grad)
        self.input_size = int(input_size)
        self.output_size = int(output_size)

        layers = []
        for i in range(int(layer_size)):
            layers.append(nn.Linear(self.input_size, self.input_size))
            layers.append(nn.ReLU(inplace=True))
        # final projection to output_size
        layers.append(nn.Linear(self.input_size, self.output_size))

        self.net = nn.Sequential(*layers)

        if self.no_grad_mode:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, x):
        """
        Forward a random tensor with the correct batch size and feature size through the network.

        x: tensor (only used for batch size, dtype, and device). If None, batch size defaults to 1.
        """
        if x is None:
            batch_size = 1
            device = torch.device("cpu")
            dtype = torch.get_default_dtype()
        else:
            batch_size = x.shape[0]
            device = x.device
            dtype = x.dtype

        rand = torch.randn(batch_size, self.input_size, device=device, dtype=dtype)

        if self.no_grad_mode:
            with torch.no_grad():
                out = self.net(rand)
        else:
            out = self.net(rand)

        return out