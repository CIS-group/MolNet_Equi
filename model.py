import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import torch_geometric.nn as geo_nn
from layer import MolNet_Layer
from torch_geometric.data import DataLoader
import math
import os
import numpy as np


def initialize_weight(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        try:
            torch.nn.init.constant_(m.bias.data, 0)
        except:
            pass
        
class MolNet_Eq_Invariance(torch.nn.Module):
    def __init__(self):
        super(MolNet_Eq_Invariance, self).__init__()
        self.scalar_embed = torch.nn.Linear(57, 128)
        self.gcn_1_A_ = MolNet_Layer(128, 128)
        self.gcn_1_B = MolNet_Layer(128, 128, True)

        self.gcn_2_A_ = MolNet_Layer(128, 128)
        self.gcn_2_B = MolNet_Layer(128, 128, True)

        self.sc_fc_1 = torch.nn.Linear(6 * 128, 128)
        self.sc_fc_2 = torch.nn.Linear(128, 128)

        self.fc_1 = torch.nn.Linear(128, 1)
        self.silu = torch.nn.SiLU()

    def forward(self, data):
        scalar, position, edge_index_A_, edge_index_B, edge_attr = (
            data.scalar,
            data.pos,
            data.edge_index_A_,
            data.edge_index_B,
            data.edge_attr
        )
        scalar = self.silu(self.scalar_embed(scalar))
        vector = torch.zeros_like(scalar)
        vector = vector[:, :, None]
        vector = vector.expand(-1, -1, 3).permute((0, 2, 1))
        scalar_A_, vector_A_ = self.gcn_1_A_(scalar, vector, position, edge_index_A_, edge_attr)
        scalar_B, vector_B = self.gcn_1_B(scalar, vector, position, edge_index_B, edge_attr)

        scalar_A_, vector_A_ = self.gcn_2_A_(scalar_A_, vector_A_, position, edge_index_A_, edge_attr)
        scalar_B, vector_B = self.gcn_2_B(scalar_B, vector_B, position, edge_index_B, edge_attr)
        

        scalar_f = torch.cat((scalar_A_, scalar_B), dim=-1)
        scalar_sum, scalar_avg, scalar_max = geo_nn.global_add_pool(scalar_f, data.batch), geo_nn.global_mean_pool(scalar_f, data.batch), geo_nn.global_max_pool(scalar_f, data.batch)
        scalar_cat = torch.cat([scalar_sum, scalar_avg, scalar_max], dim=-1)

        final = self.silu(self.sc_fc_2(self.silu(self.sc_fc_1(scalar_cat))))


        feature = self.fc_1(final)

        return feature


class MolNet_Eq_Dipole(torch.nn.Module):
    def __init__(self, useHs=False):
        super(MolNet_Eq_Dipole, self).__init__()
        if useHs:
            self.scalar_embed = torch.nn.Linear(58, 128)
        else:
            self.scalar_embed = torch.nn.Linear(57, 128)
        self.gcn_1_A_ = MolNet_Layer(128, 128)
        self.gcn_1_B = MolNet_Layer(128, 128, True)

        self.gcn_2_A_ = MolNet_Layer(128, 128)
        self.gcn_2_B = MolNet_Layer(128, 128, True)

        self.sc_fc_1 = torch.nn.Linear(2 * 128, 128, bias=False)
        self.sc_fc_2 = torch.nn.Linear(128, 128, bias=False)

        self.fc_1 = torch.nn.Linear(128, 1, bias=False)
        self.silu = torch.nn.SiLU()

    def forward(self, data):
        scalar, position, edge_index_A_, edge_index_B, edge_attr = (
            data.scalar,
            data.pos,
            data.edge_index_A_,
            data.edge_index_B,
            data.edge_attr
        )
        scalar = self.silu(self.scalar_embed(scalar))
        vector = torch.zeros_like(scalar)
        vector = vector[:, :, None]
        vector = vector.expand(-1, -1, 3).permute((0, 2, 1))
        scalar_A_, vector_A_ = self.gcn_1_A_(scalar, vector, position, edge_index_A_, edge_attr)
        scalar_B, vector_B = self.gcn_1_B(scalar, vector, position, edge_index_B, edge_attr)

        scalar_A_, vector_A_ = self.gcn_2_A_(scalar_A_, vector_A_, position, edge_index_A_, edge_attr)
        scalar_B, vector_B = self.gcn_2_B(scalar_B, vector_B, position, edge_index_B, edge_attr)
        

        total_vector = torch.cat((vector_A_, vector_B), dim=-1)
        vector_sum = geo_nn.global_add_pool(total_vector, data.batch)

        final = (self.sc_fc_2((self.sc_fc_1(vector_sum))))


        feature = self.fc_1(final)

        return feature
