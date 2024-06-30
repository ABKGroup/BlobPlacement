import torch
import torch.nn as nn

from torch_geometric.nn import HypergraphConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.norm import BatchNorm

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_skip=False, batch_norm=False):
        super().__init__()
        self.conv = HypergraphConv(in_channels, out_channels)
        self.act = nn.SiLU()
        self.use_skip = (in_channels == out_channels) and use_skip
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = BatchNorm(out_channels)

    def forward(self, x, hyperedge_index, hyperedge_weight=None):
        out = self.conv(x, hyperedge_index, hyperedge_weight)
        if self.batch_norm:
            out = self.bn(out)
        out = self.act(out)
        if self.use_skip:
            x = out + x
        return out

class Model(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Model, self).__init__()

        self.act = nn.SiLU()

        num_branches = 4
        dim_hid = 64
        dim_emb = 32

        self.conv_branches = nn.ModuleList([
            nn.ModuleList([
                ConvBlock(dim_in, dim_hid, batch_norm=True),
                ConvBlock(dim_hid, dim_hid, batch_norm=True, use_skip=True),
                ConvBlock(dim_hid, dim_emb, batch_norm=True)
            ])
            for _ in range(num_branches)
        ])

        dim_hid = 64 

        self.pred_module = nn.Sequential(
            nn.Linear(dim_emb, dim_hid),
            nn.BatchNorm1d(dim_hid),
            self.act,
            nn.Linear(dim_hid, dim_out),
        )

    def forward(self, x, hyperedge_index, batch, hyperedge_weight=None):
        x_branches = []
        for conv_branch in self.conv_branches:
            x_branch = x 
            for conv in conv_branch:
                x_branch = conv(x_branch, hyperedge_index, hyperedge_weight)
            x_branches.append(x_branch)
        x = torch.stack(x_branches, dim=1)
        x = torch.sum(x, dim=1)

        x = global_mean_pool(x, batch) 
        x = self.pred_module(x)
        
        return x
