import torch
import torch.nn as nn
import torch.nn.functional as F

class SplatFeatureEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        

