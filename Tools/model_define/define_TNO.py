#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/2/11 2:35
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @Site    :
# @File    : run_Trans.py
"""
import os
import numpy as np
import torch
import torch.nn as nn
from Utilizes.geometrics import gen_uniform_grid
import yaml
from basic.basic_layers import FcnSingle
from transformer.Transformers import SimpleTransformer, FourierTransformer


class TransBasedNeuralOperator(nn.Module):

    def __init__(self, 
                 in_dim=None,
                 out_dim=None,
                 n_hidden_b=64,
                 num_layers_b=2,
                 n_hidden_s=64,
                 num_layers_s=0,
                 yml_path=None,
                 ):

        super(TransBasedNeuralOperator, self).__init__()
        if yml_path is None:
            yml_path = find_file_in_directory(os.path.join('..'), 'transformer_config_gvrb.yml')
        with open(yml_path) as f:
            config = yaml.full_load(f)
            config = config['GVRB_2d']

        # 建立网络
        Tra_model = FourierTransformer(**config)
        hidden_b = [int(n_hidden_b)] * int(num_layers_b)
        MLP_model = FcnSingle(planes=(in_dim, *hidden_b, config['n_targets']), last_activation=True)
        if num_layers_s>0:
            hidden_s = [int(n_hidden_s)] * int(num_layers_s)
            Share_model = FcnSingle(planes=(config['n_targets'], *hidden_s, out_dim), last_activation=True)
        else:
            Share_model = nn.Linear(MLP_model.planes[-1], out_dim)



        self.branch_net = MLP_model
        self.trunc_net = Tra_model
        self.field_net = Share_model


        
    def forward(self, design, coords):
        """
        forward compute
        :param design: tensor list[(batch_size, ..., operator_dims[0]), (batch_size, ..., operator_dims[1]), ...]
        :param coords: (batch_size, ..., input_dim)
        """

        T = self.trunc_net(coords)
        B = self.branch_net(design)
        T_size = T.shape[1:-1]
        for i in range(len(T_size)):
            B = B.unsqueeze(1)
        B = torch.tile(B, [1, ] + list(T_size) + [1, ])
        feature = B * T
        F = self.field_net(feature)
        return F

def find_file_in_directory(directory, filename):
    for root, dirs, files in os.walk(directory):
        if filename in files:
            file_path = os.path.join(root, filename)
            return file_path
    return None


def train(dataloader, netmodel, device, lossfunc, optimizer, scheduler):
    """
    Args:
        data_loader: output fields at last time step
        netmodel: Network
        lossfunc: Loss function
        optimizer: optimizer
        scheduler: scheduler
    """

    _, bb = next(iter(dataloader))
    grid = gen_uniform_grid(torch.tensor(np.zeros([1, bb.shape[1], bb.shape[2], bb.shape[3]])))
    train_loss = 0
    for batch, (xx, yy) in enumerate(dataloader):
        xx = xx.to(device)
        yy = yy.to(device)
        coords = grid.tile([xx.shape[0], 1, 1, 1]).to(device)

        pred = netmodel(xx, coords)
        loss = lossfunc(pred, yy)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    scheduler.step()
    return train_loss / (batch + 1)


def valid(dataloader, netmodel, device, lossfunc):
    """
    Args:
        data_loader: input coordinates
        model: Network
        lossfunc: Loss function
    """

    _, bb = next(iter(dataloader))
    grid = gen_uniform_grid(torch.tensor(np.zeros([1, bb.shape[1], bb.shape[2], bb.shape[3]])))
    valid_loss = 0
    with torch.no_grad():
        for batch, (xx, yy) in enumerate(dataloader):
            xx = xx.to(device)
            yy = yy.to(device)
            coords = grid.tile([xx.shape[0], 1, 1, 1]).to(device)
            pred = netmodel(xx, coords)
            loss = lossfunc(pred, yy)
            valid_loss += loss.item()

    return valid_loss / (batch + 1)


def inference(dataloader, netmodel, device):
    """
    Args:
        dataloader: input coordinates
        netmodel: Network
    Returns:
        out_pred: predicted fields
    """

    _, bb = next(iter(dataloader))
    grid = gen_uniform_grid(torch.tensor(np.zeros([1, bb.shape[1], bb.shape[2], bb.shape[3]])))
    with torch.no_grad():
        xx, yy = next(iter(dataloader))
        xx = xx.to(device)
        yy = yy.to(device)
        coords = grid.tile([xx.shape[0], 1, 1, 1]).to(device)
        pred = netmodel(xx, coords)

    # equation = model.equation(u_var, y_var, out_pred)
    return xx.cpu().numpy(), yy.cpu().numpy(), pred.cpu().numpy()


if __name__ == "__main__":
    ################################################################
    # configs
    ################################################################

    name = 'TNO'

