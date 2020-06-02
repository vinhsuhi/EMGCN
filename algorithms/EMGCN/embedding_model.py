import torch
import torch.nn as nn

import numpy as np
from algorithms.EMGCN.utils import init_weight, get_act_function

# DONE

class GCN(nn.Module):
    """
    The GCN multistates block
    """
    def __init__(self, activate_function, input_dim, output_dim):
        """
        activate_function: Tanh
        input_dim: input features dimensions
        output_dim: output features dimensions
        """
        super(GCN, self).__init__()
        if activate_function is not None:
            self.activate_function = get_act_function(activate_function)
        else:
            self.activate_function = None
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        # self.linear = nn.DataParallel(self.linear)
        init_weight(self.modules(), "tanh")
    
    def forward(self, A_hat, input):
        """
        :params A_hat: adjacency matrix for this GCN layer
        :params input: input matrix for this GCN layer
        """
        # last layer we do not have weight matrix
        # if self.activate_function is not None:
        output = self.linear(input)
        
        output = torch.matmul(A_hat, output)

        # do not activate at last layer
        if self.activate_function is not None:
            output = self.activate_function(output)
        return output



class EM_GCN(nn.Module):
    """
    Training a multilayer GCN model
    """
    def __init__(self, activate_function, num_GCN_blocks, output_dim, \
                num_source_nodes, num_target_nodes, source_feats=None, target_feats=None, direct=True):
        """
        :params activate_function: Name of activation function
        :params num_GCN_blocks: Number of GCN layers of model
        :params output_dim: The number of dimensions of output
        :params num_source_nodes: Number of nodes in source graph
        :params num_target_nodes: Number of nodes in target graph
        :params source_feats: Source Initialized Features
        :params target_feats: Target Initialized Features
        :params direct: Whether to run model in direct mode
        """
        super(EM_GCN, self).__init__()
        self.num_GCN_blocks = num_GCN_blocks 
        self.direct = direct

        self.source_feats = source_feats
        self.target_feats = target_feats
        input_dim = self.source_feats.shape[1]
        self.activate_function = get_act_function(activate_function)

        self.GCNs = []
        for i in range(num_GCN_blocks):
            # Last layer is like GCN-align
            if i == num_GCN_blocks - 1:
                self.GCNs.append(GCN("", input_dim, output_dim))
            else:
                self.GCNs.append(GCN(activate_function, input_dim, output_dim))
            input_dim = self.GCNs[-1].output_dim
            
        self.GCNs = nn.ModuleList(self.GCNs)
        init_weight(self.modules(), activate_function)


    def forward_direct(self, A_hat, emb_input):
        """
        :params A_hat: adjacency matrix for this GCN layer
        :params emb_input: emb of the previous layer of initial embedding
        """
        outputs = [emb_input]
        GCN_input_i1 = emb_input
        GCN_input_i2 = emb_input
        for i in range(self.num_GCN_blocks):
            GCN_output_i1 = self.GCNs[i](A_hat, GCN_input_i1)
            GCN_output_i2 = self.GCNs[i](A_hat.t(), GCN_input_i2)
            GCN_output_i = torch.cat((GCN_output_i1, GCN_output_i2), dim=1)
            outputs.append(GCN_output_i)
            GCN_input_i1 = GCN_output_i1
            GCN_input_i2 = GCN_output_i2
        return outputs

    
    def forward_undirect(self, A_hat, emb_input):
        """
        :params A_hat: adjacency matrix for this GCN layer
        :params emb_input: emb of the previous layer of initial embedding
        """
        outputs = [emb_input]
        for i in range(self.num_GCN_blocks):
            GCN_output = self.GCNs[i](A_hat, emb_input)
            outputs.append(GCN_output)
            emb_input = GCN_output
        return outputs
    

    def forward(self, A_hat, net='s'):
        """
        Do the forward
        :params A_hat: The sparse Normalized Laplacian Matrix 
        :params net: Whether forwarding graph is source or target graph
        """
        if net == 's':
            input = self.source_feats
        else:
            input = self.target_feats
        emb_input = input.clone()

        if self.direct:
            return self.forward_direct(A_hat, emb_input)
        else:
            return self.forward_undirect(A_hat, emb_input)

