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
from torch.utils.data import DataLoader
from Utilizes.process_data import DataNormer
from basic.basic_layers import FcnSingle
from transformer.Transformers import FourierTransformer
from Utilizes.geometrics import gen_uniform_grid
from Utilizes.visual_data import MatplotlibVision, TextLogger
from itertools import chain, cycle
import matplotlib.pyplot as plt
import time
import yaml
from Demo.PakB_2d.utilizes_pakB import get_origin, PakBWeightLoss
from Demo.GVRB_2d.train_model_GVRB.model_whole_life import WorkPrj

os.chdir(r'/')
class predictor(nn.Module):

    def __init__(self, branch, trunc, supercondtion, share, super_order=None):

        super(predictor, self).__init__()

        self.order = super_order

        self.branch_net = branch
        self.trunc_net = trunc
        self.super_net = supercondtion
        self.field_net = share
        # self.field_net = nn.Linear(branch.planes[-1], field_dim)


    def forward(self, design, coords):
        """
        forward compute
        :param design: tensor list[(batch_size, ..., operator_dims[0]), (batch_size, ..., operator_dims[1]), ...]
        :param coords: (batch_size, ..., input_dim)
        """
        hole_num = design.shape[0]

        T = self.trunc_net(coords)
        B = self.branch_net(design)
        feature_list = []
        for ii in range(hole_num):
            feature_list.append(B[ii] * T)

        super_order_tmp = self.order
        for _ in range(hole_num-1):
            idx = int(super_order_tmp[0])
            F_input = torch.cat(feature_list[idx:idx+2], axis=-1)
            F_combine = self.super_net(F_input)
            # change the list and order
            feature_list.pop(idx + 1)
            feature_list.pop(idx)
            feature_list.insert(idx, F_combine)

            super_order_tmp = super_order_tmp[1:]
            super_order_tmp = [x - 1 if x > idx else x for x in super_order_tmp]

        feature_all = feature_list[0]
        sdf_all = design[0]
        for ii in range(hole_num - 1):
            sdf_all = torch.min(sdf_all, design[ii+1])
        F = self.field_net(torch.cat((feature_all, sdf_all), axis=-1))
        return F

def train(dataloader, netmodel, device, lossfunc, optimizer, scheduler):
    """
    Args:
        data_loader: output fields at last time step
        netmodel: Network
        lossfunc: Loss function
        optimizer: optimizer
        scheduler: scheduler
    """
    grid = gen_uniform_grid(torch.tensor(np.zeros([1, 128, 128, 1]))).to(device)
    train_loss = 0
    for batch, (xx, yy, ww) in enumerate(dataloader):
        xx = xx.to(device)
        coords = grid.tile([xx.shape[0], 1, 1, 1])
        xx = xx.permute(3, 0, 1, 2).unsqueeze(-1)
        yy = yy.to(device)
        pred = netmodel(xx, coords)
        loss = lossfunc(ww, pred, yy)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if batch >= batch_iter - 1:
            break

    scheduler.step()
    return train_loss / (batch + 1)


def valid(dataloader, netmodel, device, lossfunc):
    """
    Args:
        data_loader: input coordinates
        model: Network
        lossfunc: Loss function
    """
    grid = gen_uniform_grid(torch.tensor(np.zeros([1, 128, 128, 1]))).to(device)
    valid_loss = 0
    with torch.no_grad():
        for batch, (xx, yy, ww) in enumerate(dataloader):
            xx = xx.to(device)
            coords = grid.tile([xx.shape[0], 1, 1, 1])
            xx = xx.permute(3, 0, 1, 2).unsqueeze(-1)
            yy = yy.to(device)

            pred = netmodel(xx, coords)
            loss = lossfunc(ww, pred, yy)
            valid_loss += loss.item()

            if batch >= batch_iter -1:
                break

    return valid_loss / (batch + 1)


def inference(dataloader, netmodel, device):
    """
    Args:
        dataloader: input coordinates
        netmodel: Network
    Returns:
        out_pred: predicted fields
    """
    grid = gen_uniform_grid(torch.tensor(np.zeros([1, 128, 128, 1]))).to(device)
    xx = None
    yy = None# 初始化 xx
    pred = None  # 初始化 pred
    with torch.no_grad():
        for batch, (xx, yy, ww) in enumerate(dataloader):
        # xx, yy = next(iter(dataloader))
            xx = xx.to(device)
            coords = grid.tile([xx.shape[0], 1, 1, 1])
            xx = xx.permute(3, 0, 1, 2).unsqueeze(-1)

            pred = netmodel(xx, coords)
            break

    # equation = model.equation(u_var, y_var, out_pred)
    return xx.cpu().numpy(), yy.numpy(), pred.cpu().numpy()

def get_loader_pakB(input, output,
           x_normalizer=None,
           y_normalizer=None,
           ntrain=800,
           nvalid=150,
           ):
    tmp, _ = torch.min(input, dim=-1)
    weight = (tmp > 0).int()

    train_x = input[:ntrain]
    train_y = output[:ntrain]
    train_w = weight[:ntrain]
    # train_g = grids[:ntrain, ::r1]
    valid_x = input[-nvalid:]
    valid_y = output[-nvalid:]
    valid_w = weight[-nvalid:]

    train_x = x_normalizer.norm(train_x)
    valid_x = x_normalizer.norm(valid_x)

    train_y = y_normalizer.norm(train_y)
    valid_y = y_normalizer.norm(valid_y)
    #
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y, train_w),
                                               batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(valid_x, valid_y, valid_w),
                                               batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, valid_loader



if __name__ == "__main__":
    ################################################################
    # configs
    ################################################################
    name = 'TNO'
    work_path = os.path.join('Demo', 'PakB_2d', '../../work', name + '_' + str(2))
    train_path = os.path.join(work_path)
    isCreated = os.path.exists(work_path)
    if not isCreated:
        os.makedirs(work_path)
        # os.makedirs(train_path)

    # 将控制台的结果输出到log文件
    Logger = TextLogger(os.path.join(train_path, 'train.log'))

    if torch.cuda.is_available():
        Device = torch.device('cuda')
    else:
        Device = torch.device('cpu')

    # Device = torch.device('cpu')
    # design, fields = get_origin_old()
    # fields = fields[:, 0].transpose(0, 2, 3, 1)

    in_dim = 1
    out_dim = 1
    ntrain = 800
    nvalid = 150

    batch_size = 32
    batch_iter = 60
    epochs = 1001
    learning_rate = 0.001
    scheduler_step = 700
    scheduler_gamma = 0.1
    r1 = 1
    print(epochs, learning_rate, scheduler_step, scheduler_gamma)
    # #这部分应该是重采样
    # #不进行稀疏采样
    #
    #
    # ################################################################
    # # load data
    # ################################################################
    work = WorkPrj(work_path)
    x_normalizer = None
    y_normalizer = None
    train_loader_list = []
    valid_loader_list = []
    for kk, hole_num in enumerate([1,2]):
        design, fields, grids = get_origin(type='struct', hole_num=hole_num)  # 获取原始数据取原始数据

        input = design
        input = torch.tensor(input, dtype=torch.float)
        output = fields
        output = torch.tensor(output, dtype=torch.float)
        print(input.shape, output.shape)

        if x_normalizer is None:
            x_normalizer = DataNormer(input.numpy(), method='mean-std')
            y_normalizer = DataNormer(output.numpy(), method='mean-std')
            x_normalizer.save(os.path.join(work_path, 'x_norm.pkl'))  # 将normalizer保存下来
            y_normalizer.save(os.path.join(work_path, 'y_norm.pkl'))

        train_loader, valid_loader = get_loader_pakB(input, output,
                                       x_normalizer=x_normalizer,
                                       y_normalizer=y_normalizer,
                                       ntrain=ntrain,
                                       nvalid=nvalid,
                                       )

        train_loader_list.append(train_loader)
        valid_loader_list.append(valid_loader)

    train_loader = cycle(chain(train_loader_list[0], train_loader_list[1]))
    valid_loader = cycle(chain(valid_loader_list[0], valid_loader_list[1]))


    #
    #
    # ################################################################
    # #  Neural Networks
    # ################################################################
    with open(os.path.join('Demo', 'GVRB_2d', '../../data', 'configs', 'transformer_config_gvrb.yml')) as f:
        config = yaml.full_load(f)
        config = config['GVRB_2d']
    #
    # # 建立网络
    Tra_model = FourierTransformer(**config).to(Device)
    # Tra_model = SimpleTransformer(**config).to(Device)
    # FNO_model = FNO2d(in_dim=2, out_dim=config['n_targets'], modes=(16, 16), width=64, depth=4,
    #                   padding=9, activation='gelu').to(Device)
    MLP_model = FcnSingle(planes=(in_dim, 64, 64, 64, config['n_targets']), last_activation=True).to(Device)
    # Net_model = predictor(trunc=Tra_model, branch=MLP_model, field_dim=out_dim).to(Device)
    Share_model = FcnSingle(planes=(config['n_targets']+1, 64, 64, out_dim), last_activation=False).to(Device)

    super_model = FcnSingle(planes=(config['n_targets']*2, 64, 64, config['n_targets']), last_activation=False).to(Device)
    Net_model = predictor(trunc=Tra_model, branch=MLP_model, supercondtion=super_model, share=Share_model, super_order=[0,]).to(Device)
    isExist = os.path.exists(work.pth)
    log_loss = [[], []]
    if isExist:
        print(work.pth)
        checkpoint = torch.load(work.pth, map_location=Device)
        Net_model.load_state_dict(checkpoint['net_model'])
        log_loss = checkpoint['log_loss']
        Net_model.eval()


    # model_statistics = summary(Net_model, input_size=(batch_size, train_x.shape[1]), device=str(Device))
    # Logger.write(str(model_statistics))
    #
    # # 损失函数
    # Loss_func = nn.MSELoss()
    Loss_func = PakBWeightLoss(0)
    # # Loss_func = nn.SmoothL1Loss()
    # # 优化算法
    Optimizer = torch.optim.Adam(Net_model.parameters(), lr=learning_rate, betas=(0.7, 0.9),)# weight_decay=1e-7)
    # # 下降策略
    Scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    # # 可视化
    Visual = MatplotlibVision(work_path, input_name=('x', 'y'), field_name=('TEMPERATURE',))


    star_time = time.time()
    log_loss = [[], []]

    ################################################################
    # train process
    ################################################################
    # grid = get_grid()
    # grid_real = get_grid()
    # grid = gen_uniform_grid(train_y[:1]).to(Device)
    for epoch in range(epochs):

        Net_model.train()
        log_loss[0].append(train(train_loader, Net_model, Device, Loss_func, Optimizer, Scheduler))

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
            fig.savefig(os.path.join(train_path, 'log_loss.svg'))
            plt.close(fig)

        ################################################################
        # Visualization
        ################################################################
        if epoch % 100 == 0:
            train_source, train_true, train_pred = inference(train_loader, Net_model, Device)
            valid_source, valid_true, valid_pred = inference(valid_loader, Net_model, Device)

            torch.save(
                {'log_loss': log_loss, 'net_model': Net_model.state_dict(), 'optimizer': Optimizer.state_dict()},
                os.path.join(work_path, 'latest_model.pth'))

            for fig_id in range(5):
                fig, axs = plt.subplots(out_dim, 3, figsize=(18, 20), num=2)
                Visual.plot_fields_ms(fig, axs, train_true[fig_id], train_pred[fig_id], grids)
                fig.savefig(os.path.join(work_path, 'train_solution_' + str(fig_id) + '.jpg'))
                plt.close(fig)

            for fig_id in range(5):
                fig, axs = plt.subplots(out_dim, 3, figsize=(18, 20), num=3)
                Visual.plot_fields_ms(fig, axs, valid_true[fig_id], valid_pred[fig_id], grids)
                fig.savefig(os.path.join(work_path, 'valid_solution_' + str(fig_id) + '.jpg'))
                plt.close(fig)



