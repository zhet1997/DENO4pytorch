#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
# @Time    : 2022/11/27 0:15
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @Site    :
# @File    : run_Darcy_train..py.py
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Utilizes.process_data import DataNormer
from don.DeepONets import DeepONetMulti
from Utilizes.visual_data import MatplotlibVision
import matplotlib.pyplot as plt
import time
from Demo.GVRB_2d.utilizes_GVRB import get_origin

def train(dataloader, netmodel, device, lossfunc, optimizer, scheduler):
    """
    Args:
        data_loader: output fields at last time step
        netmodel: Network
        lossfunc: Loss function
        optimizer: optimizer
        scheduler: scheduler
    """

    train_loss = 0
    for batch, (f, x, u) in enumerate(dataloader):
        f = f.to(device)
        x = x.to(device)
        u = u.to(device)
        pred = netmodel([f, ], x, size_set=True)

        loss = lossfunc(pred, u)

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
    valid_loss = 0
    with torch.no_grad():
        for batch, (f, x, u) in enumerate(dataloader):
            f = f.to(device)
            x = x.to(device)
            u = u.to(device)
            pred = netmodel([f, ], x, size_set=True)

            loss = lossfunc(pred, u)
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
    with torch.no_grad():
        f, x, u = next(iter(dataloader))
        f = f.to(device)
        x = x.to(device)
        pred = netmodel([f, ], x, size_set=True)

    # equation = model.equation(u_var, y_var, out_pred)
    return x.cpu().numpy(), x.cpu().numpy(), u.numpy(), pred.cpu().numpy()


if __name__ == "__main__":
    ################################################################
    # configs
    ################################################################

    name = 'deepONet'
    work_path = os.path.join('../work', name)
    isCreated = os.path.exists(work_path)
    if not isCreated:
        os.makedirs(work_path)

    # 将控制台的结果输出到log文件
    # sys.stdout = TextLogger(os.path.join(work_path, 'train.log'), sys.stdout)

    if torch.cuda.is_available():
        Device = torch.device('cuda')
    else:
        Device = torch.device('cpu')

    design, fields, grids = get_origin()  # 获取原始数据

    in_dim = 100
    out_dim = 8

    ntrain = 5000
    nvalid = 900
    batch_size = 32
    batch_size2 = batch_size


    epochs = 1001
    learning_rate = 0.001
    scheduler_step = 800
    scheduler_gamma = 0.1



    print(epochs, learning_rate, scheduler_step, scheduler_gamma)
    r1 = 1

    ################################################################
    # load data
    ################################################################

    input = np.tile(design[:, None, :], (1, 13170, 1))
    input = torch.tensor(input, dtype=torch.float)

    # output = fields[:, 0, :, :, :].transpose((0, 2, 3, 1))
    output = fields
    output = torch.tensor(output, dtype=torch.float)

    grids = torch.tensor(grids, dtype=torch.float)
    print(input.shape, output.shape)

    train_x = input[:ntrain, ::r1]
    train_y = output[:ntrain, ::r1]
    train_g = grids[:ntrain, ::r1]
    valid_x = input[-nvalid:, ::r1]
    valid_y = output[-nvalid:, ::r1]
    valid_g = grids[-nvalid:, ::r1]

    x_normalizer = DataNormer(train_x.numpy(), method='mean-std')
    train_x = x_normalizer.norm(train_x)
    valid_x = x_normalizer.norm(valid_x)

    y_normalizer = DataNormer(train_y.numpy(), method='mean-std')
    train_y = y_normalizer.norm(train_y)
    valid_y = y_normalizer.norm(valid_y)

    g_normalizer = DataNormer(train_g.numpy(), method='mean-std')
    train_g = g_normalizer.norm(train_g)
    valid_g = g_normalizer.norm(valid_g)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_g, train_y),
                                               batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(valid_x, valid_g, valid_y),
                                               batch_size=batch_size, shuffle=False, drop_last=True)

    ################################################################
    #  Neural Networks
    ################################################################
    # 建立网络
    Net_model = DeepONetMulti(input_dim=2, operator_dims=[100, ], output_dim=8,
                              planes_branch=[64] * 3, planes_trunk=[64] * 3).to(Device)
    # 损失函数
    Loss_func = nn.MSELoss()
    # Loss_func = nn.SmoothL1Loss()
    # L1loss = nn.SmoothL1Loss()
    # 优化算法
    Optimizer = torch.optim.Adam(Net_model.parameters(), lr=learning_rate, betas=(0.7, 0.9), weight_decay=1e-4)
    # 下降策略
    Scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    # 可视化
    Visual = MatplotlibVision(work_path, input_name=('x', 'y'), field_name=('p', 't', 'rho', 'alf', 'v'))

    star_time = time.time()
    log_loss = [[], []]

    ################################################################
    # train process
    ################################################################

    # 生成网格文件

    for epoch in range(epochs):

        Net_model.train()
        log_loss[0].append(train(train_loader, Net_model, Device, Loss_func, Optimizer, Scheduler))

        Net_model.eval()
        log_loss[1].append(valid(valid_loader, Net_model, Device, Loss_func))
        print('epoch: {:6d}, lr: {:.3e}, train_step_loss: {:.3e}, valid_step_loss: {:.3e}, cost: {:.2f}'.
              format(epoch, Optimizer.param_groups[0]['lr'], log_loss[0][-1], log_loss[1][-1], time.time() - star_time))

        star_time = time.time()

        if epoch > 0 and epoch % 20 == 0:
            fig, axs = plt.subplots(1, 1, figsize=(15, 8), num=1)
            Visual.plot_loss(fig, axs, np.arange(len(log_loss[0])), np.array(log_loss)[0, :], 'train_step')
            Visual.plot_loss(fig, axs, np.arange(len(log_loss[0])), np.array(log_loss)[1, :], 'valid_step')
            fig.suptitle('training loss')
            fig.savefig(os.path.join(work_path, 'log_loss.svg'))
            plt.close(fig)

        ################################################################
        # Visualization
        ################################################################
        if epoch > 0 and epoch % 100 == 0:

            # print('epoch: {:6d}, lr: {:.3e}, eqs_loss: {:.3e}, bcs_loss: {:.3e}, cost: {:.2f}'.
            #       format(epoch, learning_rate, log_loss[-1][0], log_loss[-1][1], time.time()-star_time))
            train_source, train_coord, train_true, train_pred = inference(train_loader, Net_model, Device)
            valid_source, valid_coord, valid_true, valid_pred = inference(valid_loader, Net_model, Device)

            torch.save({'log_loss': log_loss, 'net_model': Net_model.state_dict(), 'optimizer': Optimizer.state_dict()},
                       os.path.join(work_path, 'latest_model.pth'))

            # train_true = train_true.reshape([train_true.shape[0], 64, 64, out_dim])
            # train_pred = train_pred.reshape([train_pred.shape[0], 64, 64, out_dim])
            # valid_true = valid_true.reshape([valid_true.shape[0], 64, 64, out_dim])
            # valid_pred = valid_pred.reshape([valid_pred.shape[0], 64, 64, out_dim])
            #
            # for fig_id in range(5):
            #     fig, axs = plt.subplots(out_dim, 3, figsize=(18, 20), num=2)
            #     Visual.plot_fields_ms(fig, axs, train_true[fig_id], train_pred[fig_id], grid)
            #     fig.savefig(os.path.join(work_path, 'train_solution_' + str(fig_id) + '.jpg'))
            #     plt.close(fig)
            #
            # for fig_id in range(5):
            #     fig, axs = plt.subplots(out_dim, 3, figsize=(18, 20),num=3)
            #     Visual.plot_fields_ms(fig, axs, valid_true[fig_id], valid_pred[fig_id], grid)
            #     fig.savefig(os.path.join(work_path, 'valid_solution_' + str(fig_id) + '.jpg'))
            #     plt.close(fig)

