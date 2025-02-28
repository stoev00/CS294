import os
import os.path as osp
import warnings
from math import pi as PI
from typing import Optional
import ase
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Embedding, Linear, ModuleList, Sequential
from torch_scatter import scatter
from torch_geometric.nn import radius_graph

class SchNet(torch.nn.Module):
    r"""The continuous-filter convolutional neural network SchNet from the
    `"SchNet: A Continuous-filter Convolutional Neural Network for Modeling
    Quantum Interactions" <https://arxiv.org/abs/1706.08566>`_ paper that uses
    the interactions blocks of the form

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \odot
        h_{\mathbf{\Theta}} ( \exp(-\gamma(\mathbf{e}_{j,i} - \mathbf{\mu}))),

    here :math:`h_{\mathbf{\Theta}}` denotes an MLP and
    :math:`\mathbf{e}_{j,i}` denotes the interatomic distances between atoms.

    Args:
        hidden_channels (int, optional): Hidden embedding size.
            (default: :obj:`128`)
        num_filters (int, optional): The number of filters to use.
            (default: :obj:`128`)
        num_interactions (int, optional): The number of interaction blocks.
            (default: :obj:`6`)
        num_gaussians (int, optional): The number of gaussians :math:`\mu`.
            (default: :obj:`50`)
        cutoff (float, optional): Cutoff distance for interatomic interactions.
            (default: :obj:`10.0`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            collect for each node within the :attr:`cutoff` distance.
            (default: :obj:`32`)
        readout (string, optional): Whether to apply :obj:`"add"` or
            :obj:`"mean"` global aggregation. (default: :obj:`"add"`)
        dipole (bool, optional): If set to :obj:`True`, will use the magnitude
            of the dipole moment to make the final prediction, *e.g.*, for
            target 0 of :class:`torch_geometric.datasets.QM9`.
            (default: :obj:`False`)
        mean (float, optional): The mean of the property to predict.
            (default: :obj:`None`)
        std (float, optional): The standard deviation of the property to
            predict. (default: :obj:`None`)
        atomref (torch.Tensor, optional): The reference of single-atom
            properties.
            Expects a vector of shape :obj:`(max_atomic_number, )`.
    """

    def __init__(self, hidden_channels: int = 128, num_filters: int = 128,
                 num_interactions: int = 6, num_gaussians: int = 50,
                 cutoff: float = 5.0, max_num_neighbors: int = 32,
                 readout: str = 'add', dipole: bool = False,
                 mean: Optional[float] = None, std: Optional[float] = None,
                 atomref: Optional[torch.Tensor] = None):
        super().__init__()

        

        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.readout = readout
        self.dipole = dipole
        self.readout = 'add' if self.dipole else self.readout
        self.mean = mean
        self.std = std
        self.scale = None

        atomic_mass = torch.from_numpy(ase.data.atomic_masses)

        self.embedding = Embedding(100, hidden_channels)
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, num_gaussians,
                                     num_filters, cutoff)
            self.interactions.append(block)

        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.act = ShiftedSoftplus()
        self.lin2 = Linear(hidden_channels // 2, 1)
        self.reset_parameters()


    def reset_parameters(self):
        self.embedding.reset_parameters()
        for interaction in self.interactions:
            interaction.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, z, pos, batch=None):
        """"""
        assert z.dim() == 1 and z.dtype == torch.long
        batch = torch.zeros_like(z) if batch is None else batch
        pos.requires_grad = True

        h = self.embedding(z) #TODO: compute initial node representations
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch, max_num_neighbors=self.max_num_neighbors)
        
        edge_weight = torch.norm(pos[edge_index[1]]-pos[edge_index[0]], dim= -1) #TODO: compute pairwise distances
        edge_attr = self.distance_expansion(edge_weight) #TODO: compute radial basis expansion of pairwise distances

        for interaction in self.interactions:
            #TODO: call interaction layers
            h = interaction.forward(h, edge_index, edge_weight, edge_attr)

        #linear projection to obtain atomwise energies
        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)
        energy_v = scatter(h, batch, dim=0, reduce="sum") #TODO: compute global energies per molecule
        print(energy_v.requires_grad)
        energy = torch.sum(energy_v)

        #forces = torch.autograd.grad(torch.sum(energy_v), pos)#TODO: compute forces
        forces = torch.autograd.grad(torch.sum(energy_v), pos, create_graph=True)[0]

        return energy, forces


class InteractionBlock(torch.nn.Module):
    def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff):
        super().__init__()
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.conv = CFConv(hidden_channels, hidden_channels, num_filters,
                           self.mlp, cutoff)
        self.act = ShiftedSoftplus()
        self.lin = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[2].bias.data.fill_(0)
        self.conv.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)

    def forward(self, h, edge_index, edge_weight, edge_attr):
        h = self.conv(h, edge_index, edge_weight, edge_attr) #TODO: fill in this line
        h = self.act(h)
        h = self.lin(h)
        return h


class CFConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_filters, nn, cutoff):
        super().__init__()
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, out_channels)
        self.nn = nn
        self.cutoff = cutoff
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, h, edge_index, edge_weight, edge_attr):
        h = self.lin1(h)
        c = 0.5 * (torch.cos( (edge_weight) / self.cutoff * torch.pi) + 1) #TODO: transform edge weights to [0,1]
        W = self.nn(edge_attr)
        message = c.unsqueeze(-1) * (W * h[edge_index[0]])
        #print(message)
        #print(h)
        #print(edge_index[1])
        #print(h[edge_index[1]].to(torch.int64))
        agg = scatter(message, edge_index[1], dim=0, reduce="sum")
        h =  h + agg #TODO: perform graph convolution
        h = self.lin2(h)
        return h
        

class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.gamma = 0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset) #sets the attribute self.offset = offset

    def forward(self, dist):
        #TODO: implement this function
        RBFk = torch.exp(-self.gamma * ((dist.unsqueeze(-1)) - self.offset) ** 2)
        return RBFk
    
class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift
