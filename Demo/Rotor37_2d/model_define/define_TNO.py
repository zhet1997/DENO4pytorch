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
import paddle
import paddle.nn as nn
from Utilizes.geometrics import gen_uniform_grid
import yaml
from basic.basic_layers import FcnSingle
from transformer.Transformers import SimpleTransformer, FourierTransformer


class TransBasedNeuralOperator(nn.Layer):

    def __init__(self, 
                 in_dim=None,
                 out_dim=None
                 ):

        super(TransBasedNeuralOperator, self).__init__()
        yml_path = find_file_in_directory(os.path.join('..'), 'transformer_config.yml')
        with open(yml_path) as f:
            config = yaml.full_load(f)
            config = config['Rotor37_2d']

        # 建立网络
        Tra_model = FourierTransformer(**config)
        MLP_model = FcnSingle(planes=(in_dim, 64, 64, config['n_targets']), last_activation=True)

        self.branch_net = MLP_model
        self.trunc_net = Tra_model
        self.field_net = nn.Linear(self.branch_net.planes[-1], out_dim)

        
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
        B = paddle.tile(B, [1, ] + list(T_size) + [1, ])
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
    grid = gen_uniform_grid(paddle.to_tensor(np.zeros([1, 64, 64, 5])))
    train_loss = 0
    for batch, (xx, yy) in enumerate(dataloader):
        coords = grid.tile([xx.shape[0], 1, 1, 1])

        pred = netmodel(xx, coords)
        loss = lossfunc(pred, yy)

        optimizer.clear_grad()
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
    grid = gen_uniform_grid(paddle.to_tensor(np.zeros([1, 64, 64, 5])))
    valid_loss = 0
    with paddle.no_grad():
        for batch, (xx, yy) in enumerate(dataloader):
            coords = grid.tile([xx.shape[0], 1, 1, 1])
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
    grid = gen_uniform_grid(paddle.to_tensor(np.zeros([1, 64, 64, 5])))
    with paddle.no_grad():
        xx, yy = next(iter(dataloader))
        coords = grid.tile([xx.shape[0], 1, 1, 1])
        pred = netmodel(xx, coords)

    # equation = model.equation(u_var, y_var, out_pred)
    return xx.cpu().numpy(), yy.numpy(), pred.cpu().numpy()


if __name__ == "__main__":
    ################################################################
    # configs
    ################################################################

    name = 'TNO'

