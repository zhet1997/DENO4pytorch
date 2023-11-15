
# -*- coding: utf-8 -*-
"""
# @copyright (c) 2023 Baidu.com, Inc. Allrights Reserved
@Time ： 2023/4/17 22:06
@Author ： Liu Tianyuan (liutianyuan02@baidu.com)
@Site ：run_FNO.py
@File ：run_FNO.py
"""
import os
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
print(torch.cuda.device_count())
print(torch.cuda.is_available())
import torch.nn as nn
from torch.utils.data import DataLoader
# from torchinfo import summary
# from torchsummary import summary
from fno.FNOs import FNO2d

from Utilizes.visual_data import MatplotlibVision
from Utilizes.process_data import DataNormer

import matplotlib.pyplot as plt
import time

from utilizes_rotor37 import get_grid, get_origin_GVRB


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
    return torch.cat((gridx, gridy), dim=-1).to(x.device)


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
        gd = feature_transform(xx)

        pred = netmodel(xx, gd)
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
            gd = feature_transform(xx)

            pred = netmodel(xx, gd)
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
        gd = feature_transform(xx)
        pred = netmodel(xx, gd)

    # equation = model.equation(u_var, y_var, out_pred)
    return xx.cpu().numpy(), gd.cpu().numpy(), yy.numpy(), pred.cpu().numpy()


if __name__ == "__main__":
    ################################################################
    # configs
    ################################################################
    grid = get_grid(GV_RB=True, grid_num=128)
    for mode in [8]:

        name = 'FNO_' + str(mode)
        work_path = os.path.join('work_FNO3000_1', name)
        isCreated = os.path.exists(work_path)
        if not isCreated:
            os.makedirs(work_path)

        # 将控制台的结果输出到log文件
        # sys.stdout = TextLogger(os.path.join(work_path, 'train.log'), sys.stdout)
        #  torch.cuda.set_device(1)

        if torch.cuda.is_available():
            Device = torch.device('cuda')
        else:
            Device = torch.device('cpu')

        # design, fields = get_origin()
        # design, fields = get_origin(quanlityList=["Static Pressure", "Static Temperature",
        #                                           "DensityFlow",
        #                                           'Relative Total Pressure', 'Relative Total Temperature'
        #                                           ])  # 获取原始数据
        design, fields = get_origin_GVRB()

        in_dim = 96
        out_dim = 4
        ntrain = 4600
        nvalid = 400

        # modes = (10, 10)
        modes = (4, 4)
        width = 128
        depth = 4
        steps = 1
        padding = 8
        dropout = 0.5

        batch_size = 32
        epochs = 300
        learning_rate = 0.001
        scheduler_step = 280
        scheduler_gamma = 0.1

        print(epochs, learning_rate, scheduler_step, scheduler_gamma)

        # r1 = 1
        # r2 = 1
        # s1 = int(((64 - 1) / r1) + 1)
        # s2 = int(((64 - 1) / r2) + 1)

        ################################################################
        # load data
        ################################################################

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

        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y),
                                                   batch_size=batch_size, shuffle=True, drop_last=True)
        valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(valid_x, valid_y),
                                                   batch_size=batch_size, shuffle=False, drop_last=True)

        ################################################################
        # Neural Networks
        ################################################################

        # 建立网络

        Net_model = FNO2d(in_dim=in_dim, out_dim=out_dim, modes=modes, width=width, depth=depth, steps=steps,
                          padding=padding, activation='gelu').to(Device)

        # 损失函数
        Loss_func = nn.MSELoss()
        # Loss_func = nn.SmoothL1Loss()
        # Loss_func = FieldsLpLoss(size_average=False)
        # L1loss = nn.SmoothL1Loss()
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

        for epoch in range(epochs):

            Net_model.train()
            log_loss[0].append(train(train_loader, Net_model, Device, Loss_func, Optimizer, Scheduler))

            Net_model.eval()
            log_loss[1].append(valid(valid_loader, Net_model, Device, Loss_func))
            print('epoch: {:6d}, lr: {:.3e}, train_step_loss: {:.3e}, valid_step_loss: {:.3e}, cost: {:.2f}'.
                  format(epoch, Optimizer.param_groups[0]['lr'], log_loss[0][-1], log_loss[1][-1], time.time() - star_time))

            star_time = time.time()

            if epoch > 0 and epoch % 5 == 0:
                fig, axs = plt.subplots(1, 1, figsize=(15, 8), num=1)
                Visual.plot_loss(fig, axs, np.arange(len(log_loss[0])), np.array(log_loss)[0, :], 'train_step')
                Visual.plot_loss(fig, axs, np.arange(len(log_loss[0])), np.array(log_loss)[1, :], 'valid_step')
                fig.suptitle('training loss')
                fig.savefig(os.path.join(work_path, 'log_loss.svg'))
                plt.close(fig)

            ################################################################
            # Visualization
            ################################################################

            if epoch > 0 and epoch % 10 == 0:
                train_coord, train_grid, train_true, train_pred = inference(train_loader, Net_model, Device)
                valid_coord, valid_grid, valid_true, valid_pred = inference(valid_loader, Net_model, Device)

                torch.save({'log_loss': log_loss, 'net_model': Net_model.state_dict(), 'optimizer': Optimizer.state_dict()},
                           os.path.join(work_path, 'latest_model.pth'))

                for fig_id in range(5):
                    fig, axs = plt.subplots(out_dim, 3, figsize=(18, 20), num=2)
                    Visual.plot_fields_ms(fig, axs, train_true[fig_id], train_pred[fig_id], grid)
                    fig.savefig(os.path.join(work_path, 'train_solution_' + str(fig_id) + '.jpg'))
                    plt.close(fig)
                #
                for fig_id in range(5):
                    fig, axs = plt.subplots(out_dim, 3, figsize=(18, 20), num=3)
                    Visual.plot_fields_ms(fig, axs, valid_true[fig_id], valid_pred[fig_id], grid)
                    fig.savefig(os.path.join(work_path, 'valid_solution_' + str(fig_id) + '.jpg'))
                    plt.close(fig)
                #