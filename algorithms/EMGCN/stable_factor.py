import torch
import torch.nn as nn

import numpy as np
from torch.nn import init

# DONE

class StableFactor(nn.Module):
    """
    Stable factor following each node
    """
    def __init__(self, num_source_nodes, num_target_nodes, cuda=True):
        """
        :param num_source_nodes: Number of nodes in source graph
        :param num_target_nodes: Number of nodes in target graph
        """
        super(StableFactor, self).__init__()
        self.alpha_source = torch.ones(num_source_nodes)
        self.alpha_target = torch.ones(num_target_nodes)
        if cuda:
            self.alpha_source = self.alpha_source.cuda()
            self.alpha_target = self.alpha_target.cuda()
        self.use_cuda = cuda
    

    def forward(self, A_hat, net='s'):
        """
        :param A_hat: Normalized Laplacian Matrix
        :param net: whether graph considering is source or target graph.
        """
        if net=='s':
            self.alpha = self.alpha_source
        else:
            self.alpha = self.alpha_target
        alpha_colum = self.alpha.reshape(len(self.alpha), 1)
        if self.use_cuda:
            alpha_colum = alpha_colum.cuda()
        A_hat_new = (alpha_colum * (A_hat.mul(alpha_colum)).t()).t()
        return A_hat_new
