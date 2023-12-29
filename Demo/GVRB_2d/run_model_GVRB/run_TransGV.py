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
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.cuda.is_available()
import torch.nn as nn
from torch.utils.data import DataLoader
# from torchinfo import summary
from Utilizes.process_data import DataNormer
from transformer.Transformers import FourierTransformer
from Utilizes.visual_data import MatplotlibVision, TextLogger

import matplotlib.pyplot as plt
import time

import sys
import yaml
from utilizes_rotor37 import get_grid, get_origin_GVRB
from Tools.post_process.load_model import get_true_pred
from model_whole_life import WorkPrj

def feature_transform(x):
    """
    Args:
        x: input coordinates
    Returns:
        res: input transform
    """
    shape = x.shape
    batchsize, size_x, size_y = shape[0], shape[1], shape[2]
    gridx = torch.linspace(0, 1, size_x, dtype=torch.float32)
    gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
    gridy = torch.linspace(0, 1, size_y, dtype=torch.float32)
    gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])

    edge = torch.ones((x.shape[0], 1))
    return torch.cat((gridx, gridy), dim=-1).to(x.device), edge.to(x.device)


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
    for batch, (xx, yy) in enumerate(dataloader):
        xx = xx.to(device)
        yy = yy.to(device)
        grid, edge = feature_transform(xx)

        # pred = netmodel(xx, grid, edge, grid)['preds']
        pred = netmodel(xx, grid, edge, grid)
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
    valid_loss = 0
    with torch.no_grad():
        for batch, (xx, yy) in enumerate(dataloader):
            xx = xx.to(device)
            yy = yy.to(device)
            grid, edge = feature_transform(xx)

            # pred = netmodel(xx, grid, edge, grid)['preds']
            pred = netmodel(xx, grid, edge, grid)
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
    with torch.no_grad():
        xx, yy = next(iter(dataloader))
        xx = xx.to(device)
        grid, edge = feature_transform(xx)
        # pred = netmodel(xx, grid, edge, grid)['preds']
        pred = netmodel(xx, grid, edge, grid)

    # equation = model.equation(u_var, y_var, out_pred)
    return xx.cpu().numpy(), grid.cpu().numpy(), yy.numpy(), pred.cpu().numpy()


if __name__ == "__main__":
    ################################################################
    # configs
    ################################################################
    # for mode in [8,10,12,14,16]:

        # name = 'Transformer_' + str(mode)

        name = 'Transformer'
        work_path = os.path.join('work_Trans5000_3', name)
        isCreated = os.path.exists(work_path)
        if not isCreated:
            os.makedirs(work_path)

        # 将控制台的结果输出到log文件
        sys.stdout = TextLogger(os.path.join(work_path, 'train.log'), sys.stdout)

        if torch.cuda.is_available():
            Device = torch.device('cuda:0')
        else:
            Device = torch.device('cpu')


        in_dim = 96
        out_dim = 8
        ntrain = 4000
        nvalid = 1000

        # modes = (12, 12)
        # width = 32
        # depth = 4
        # steps = 1
        # padding = 9
        # dropout = 0.0

        batch_size = 32
        epochs = 1001
        learning_rate = 0.001
        scheduler_step = 800
        scheduler_gamma = 0.1

        print(epochs, learning_rate, scheduler_step, scheduler_gamma)

        #不进行重采样
        # r_train = 1
        # h_train = int(((64 - 1) / r_train) + 1)
        # s_train = h_train
        #
        # r_valid = 1
        # h_valid = int(((64 - 1) / r_valid) + 1)
        # s_valid = h_valid

        ################################################################
        # load data
        ################################################################
        # design, fields = get_origin_GVRB()
        design, fields = get_origin_GVRB(quanlityList=["Static Pressure", "Static Temperature", "Density",
                                                       "Vx", "Vy", "Vz",
                                                       'Relative Total Temperature',
                                                       'Absolute Total Temperature'])
        input = np.tile(design[:, None, None, :], (1, 128, 128, 1))
        input = torch.tensor(input, dtype=torch.float)

        # output = fields[:, 0, :, :, :].transpose((0, 2, 3, 1))
        output = fields
        output = torch.tensor(output, dtype=torch.float)

        print(input.shape, output.shape)

        train_x = input[:ntrain, :, :]
        train_y = output[:ntrain, :, :]
        valid_x = input[ntrain:ntrain + nvalid, :, :]
        valid_y = output[ntrain:ntrain + nvalid, :, :]

        x_normalizer = DataNormer(train_x.numpy(), method='mean-std')
        train_x = x_normalizer.norm(train_x)
        valid_x = x_normalizer.norm(valid_x)

        y_normalizer = DataNormer(train_y.numpy(), method='mean-std')
        train_y = y_normalizer.norm(train_y)
        valid_y = y_normalizer.norm(valid_y)

        x_normalizer.save(os.path.join(work_path, 'x_norm.pkl'))  # 将normalizer保存下来
        y_normalizer.save(os.path.join(work_path, 'y_norm.pkl'))

        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y),
                                                   batch_size=batch_size, shuffle=True, drop_last=True)
        valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(valid_x, valid_y),
                                                   batch_size=batch_size, shuffle=False, drop_last=True)

        ################################################################
        #  Neural Networks
        ################################################################
        with open(os.path.join('../data/configs/transformer_config_8.yml')) as f:
            config = yaml.full_load(f)
            config = config['GV_RB']

        # config['fourier_modes'] = mode

        # 建立网络
        Net_model = FourierTransformer(**config).to(Device)
        # summary(Net_model, input_size=(batch_size, train_x.shape[1]), device=Device)

        # 损失函数
        Loss_func = nn.MSELoss()
        # Loss_func = nn.SmoothL1Loss()
        # 优化算法
        Optimizer = torch.optim.Adam(Net_model.parameters(), lr=learning_rate, betas=(0.7, 0.9), weight_decay=1e-4)
        # 下降策略
        Scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
        # 可视化
        Visual = MatplotlibVision(work_path, input_name=('x', 'y'), field_name=('p', 't', 'rho', 'alf'))

        star_time = time.time()
        log_loss = [[], []]

        ################################################################
        # train process
        ################################################################
        grid = get_grid(GV_RB=True, grid_num=128)
        for epoch in range(epochs):

            Net_model.train()
            log_loss[0].append(train(train_loader, Net_model, Device, Loss_func, Optimizer, Scheduler))
            # log_loss[0].append(train(train_loader, Net_model, Device, Loss_func, Optimizer))

            Net_model.eval()
            log_loss[1].append(valid(valid_loader, Net_model, Device, Loss_func))
            print('epoch: {:6d}, lr: {:.3e}, train_step_loss: {:.3e}, valid_step_loss: {:.3e}, cost: {:.2f}'.
                  format(epoch, learning_rate, log_loss[0][-1], log_loss[1][-1], time.time() - star_time))

            star_time = time.time()

            if epoch > 0 and epoch % 10 == 0:
                fig, axs = plt.subplots(1, 1, figsize=(15, 8), num=1)
                Visual.plot_loss(fig, axs, np.arange(len(log_loss[0])), np.array(log_loss)[0, :], 'train_step')
                Visual.plot_loss(fig, axs, np.arange(len(log_loss[0])), np.array(log_loss)[1, :], 'valid_step')
                fig.suptitle('training loss')
                fig.savefig(os.path.join(work_path, 'log_loss.svg'))
                plt.close(fig)

            ################################################################
            # Visualization
            ################################################################

            if epoch % 100 == 0:
                # print('epoch: {:6d}, lr: {:.3e}, eqs_loss: {:.3e}, bcs_loss: {:.3e}, cost: {:.2f}'.
                #       format(epoch, learning_rate, log_loss[-1][0], log_loss[-1][1], time.time()-star_time))
                # train_source, train_coord, train_true, train_pred = inference(train_loader, Net_model, Device)
                # valid_source, valid_coord, valid_true, valid_pred = inference(valid_loader, Net_model, Device)

                torch.save({'log_loss': log_loss, 'net_model': Net_model.state_dict(), 'optimizer': Optimizer.state_dict()},
                           os.path.join(work_path, 'latest_model.pth'))

                # for fig_id in range(5):
                #     fig, axs = plt.subplots(out_dim, 3, figsize=(18, 25), num=2)
                #     Visual.plot_fields_ms(fig, axs, train_true[fig_id], train_pred[fig_id], grid)
                #     fig.savefig(os.path.join(work_path, 'train_solution_' + str(fig_id) + '.jpg'))
                #     plt.close(fig)
                #
                # for fig_id in range(5):
                #     fig, axs = plt.subplots(out_dim, 3, figsize=(18, 25), num=3)
                #     Visual.plot_fields_ms(fig, axs, valid_true[fig_id], valid_pred[fig_id], grid)
                #     fig.savefig(os.path.join(work_path, 'valid_solution_' + str(fig_id) + '.jpg'))
                #     plt.close(fig)
                work = WorkPrj(os.path.join("D:\WQN\CODE\DENO4pytorch-main\Demo\GV_RB\work_Trans5000_3",name))
                # train_loader, valid_loader, _, _ = loaddata_Sql(name, **work.config("Basic"))

                for type in ["valid", "train"]:
                    if type == "valid":
                        true, pred = get_true_pred(valid_loader, Net_model, inference, Device,
                                                   name=name, iters=1, alldata=False)
                    elif type == "train":
                        true, pred = get_true_pred(train_loader, Net_model, inference, Device,
                                                   name=name, iters=1, alldata=False)

                    true = y_normalizer.back(true)
                    pred = y_normalizer.back(pred)

                    grid = get_grid(GV_RB=True, grid_num=128)

                    quanlityList = ["Static Pressure", "Static Temperature", "Density",
                                    "Vx", "Vy", "Vz",
                                    'Relative Total Temperature',
                                    'Absolute Total Temperature']


                    Visual = MatplotlibVision(work_path, input_name=('x', 'y'), field_name=(
                    "Static Pressure", "Static Temperature", "Density",
                                                   "Vx", "Vy", "Vz",
                                                   'Relative Total Temperature',
                                                   'Absolute Total Temperature'))
                    for fig_id in range(5):
                        fig, axs = plt.subplots(8, 3, figsize=(30, 40), num=2)
                        Visual.plot_fields_ms(fig, axs, true[fig_id], pred[fig_id], grid)
                        fig.savefig(os.path.join(work_path, 'solution_' + type + "_" + str(fig_id) + '.jpg'))
                        plt.close(fig)