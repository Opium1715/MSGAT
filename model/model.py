import torch
from torch import nn


class MSGAT(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
