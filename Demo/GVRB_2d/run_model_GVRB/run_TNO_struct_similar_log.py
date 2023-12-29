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
torch.set_default_dtype(torch.float64)
import torch.nn as nn
from torch.utils.data import DataLoader
# from torchsummary import summary
from Utilizes.process_data import DataNormer
from Models.basic.basic_layers import FcnSingle
from Models.transformer.Transformers import FourierTransformer
from Utilizes.geometrics import gen_uniform_grid
from Utilizes.visual_data import MatplotlibVision, TextLogger

import matplotlib.pyplot as plt
import time
import yaml
from Demo.GVRB_2d.utilizes_GVRB import get_origin, GVRBWeightLoss
from Demo.GVRB_2d.train_model_GVRB.model_whole_life import WorkPrj
from Tools.post_process.post_CFD import cfdPost_2d
torch.set_default_dtype(torch.float64)
class predictor(nn.Module):

    def __init__(self, branch, trunc, share, field_dim):

        super(predictor, self).__init__()

        self.branch_net = branch
        self.trunc_net = trunc
        self.field_net = share
        # self.field_net = nn.Linear(branch.planes[-1], field_dim)


    def forward(self, design, coords):
        """
        forward compute
        :param design: tensor list[(batch_size, ..., operator_dims[0]), (batch_size, ..., operator_dims[1]), ...]
        :param coords: (batch_size, ..., input_dim)
        """

        T = self.trunc_net(coords)
        B = self.branch_net(design)
        T_size = T.shape[1:-1]
        C = design[:,-4:]
        for i in range(len(T_size)):
            B = B.unsqueeze(1)
            C = C.unsqueeze(1)
        B = torch.tile(B, [1, ] + list(T_size) + [1, ])
        C = torch.tile(C, [1, ] + list(T_size) + [1, ])
        feature = B * T
        F = self.field_net( torch.cat((feature,C), axis=-1))
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
    grid = gen_uniform_grid(torch.tensor(np.zeros([1, 64, 128, 6]))).to(device)
    train_loss = 0
    for batch, (xx, yy) in enumerate(dataloader):
        xx = xx.to(device)
        yy = yy.to(device)
        coords = grid.tile([xx.shape[0], 1, 1, 1])

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
    grid = gen_uniform_grid(torch.tensor(np.zeros([1, 64, 128, 6]))).to(device)
    valid_loss = 0
    with torch.no_grad():
        for batch, (xx, yy) in enumerate(dataloader):
            xx = xx.to(device)
            yy = yy.to(device)
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
    grid = gen_uniform_grid(torch.tensor(np.zeros([1, 64, 128, 6]))).to(device)
    with torch.no_grad():
        xx, yy = next(iter(dataloader))
        xx = xx.to(device)
        coords = grid.tile([xx.shape[0], 1, 1, 1])
        pred = netmodel(xx, coords)

    # equation = model.equation(u_var, y_var, out_pred)
    return xx.cpu().numpy(), yy.numpy(), pred.cpu().numpy()

if __name__ == "__main__":
    ################################################################
    # configs
    ################################################################


    name = 'TNO'
    work_path = os.path.join('../work', name + '_' + str(33) + '_log')
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

    design, fields, grids = get_origin(type='struct', realpath=r'E:\\WQN\\CODE\\DENO4pytorch\Demo\GVRB_2d\data/',
                                       quanlityList = ["Static Pressure", "Static Temperature", "Density",
                                                        "Vx", "Vy", "Vz"]
                                       )  # 获取原始数据取原始数据

    in_dim = 100
    out_dim = 6
    ntrain = 3000
    nvalid = 900

    batch_size = 32
    epochs = 500
    # learning_rate = 0.001
    # scheduler_step = 150
    # scheduler_gamma = 0.5

    learning_rate = 0.0002
    scheduler_step = 300
    scheduler_gamma = 0.5
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
    input = design
    input = torch.tensor(input, dtype=torch.float)

    # output = fields[:, 0, :, :, :].transpose((0, 2, 3, 1))
    output = fields
    output = torch.tensor(output, dtype=torch.float)

    print(input.shape, output.shape)
    #
    train_x = input[:ntrain]
    train_y = output[:ntrain]
    # train_g = grids[:ntrain, ::r1]
    valid_x = input[-nvalid:]
    valid_y = output[-nvalid:]
    # valid_g = grids[-nvalid:, ::r1]
    #
    x_normalizer = DataNormer(train_x.numpy(), method='log')
    x_normalizer.dim_change(2)
    x_normalizer.save(os.path.join(work_path, 'x_norm.pkl'))  # 将normalizer保存下来
    train_x = x_normalizer.norm(train_x)
    valid_x = x_normalizer.norm(valid_x)

    x_normalizer_bc = x_normalizer
    x_normalizer_bc.shrink(slice(96,100,1))

    y_normalizer = DataNormer(train_y.numpy(), method='log')
    y_normalizer.dim_change(2)
    train_y = y_normalizer.norm(train_y)
    valid_y = y_normalizer.norm(valid_y)
    #

    y_normalizer.save(os.path.join(work_path, 'y_norm.pkl'))

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y),
                                               batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(valid_x, valid_y),
                                               batch_size=batch_size, shuffle=False, drop_last=True)
    #test
    #
    # ################################################################
    # #  Neural Networks
    # ################################################################
    with open(os.path.join('../data/configs/transformer_config_gvrb.yml')) as f:
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
    Share_model = FcnSingle(planes=(config['n_targets']+4, 64, 64, out_dim), last_activation=False).to(Device)
    Net_model = predictor(trunc=Tra_model, branch=MLP_model, share=Share_model, field_dim=out_dim).to(Device)

    isExist = os.path.exists(work.pth)
    log_loss = [[], [], [], [], []]
    if isExist:
        print(work.pth)
        checkpoint = torch.load(work.pth, map_location=Device)
        Net_model.load_state_dict(checkpoint['net_model'])
        log_loss = checkpoint['log_loss']
        Net_model.eval()

    # torch.save(Net_model, os.path.join(work_path, 'final_model.pth'))
    # model_statistics = summary(Net_model, input_size=(batch_size, train_x.shape[1]), device=str(Device))
    # Logger.write(str(model_statistics))
    #
    # # 损失函数
    # Loss_func = nn.MSELoss()
    Loss_func = GVRBWeightLoss(4, 10, 71)
    # # Loss_func = nn.SmoothL1Loss()
    # # 优化算法
    Optimizer = torch.optim.Adam(Net_model.parameters(), lr=learning_rate, betas=(0.7, 0.9))#, weight_decay=1e-7)
    # # 下降策略
    Scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    # # 可视化
    Visual = MatplotlibVision(work_path, input_name=('x', 'y'), field_name=('ps', 'ts', 'rho', 'vx', 'vy', 'vz'))


    star_time = time.time()
    inputdict = {'Static Pressure': 0,
                 'Static Temperature': 1,
                 'Density': 2,
                 'Vx': 3,
                 'Vy': 4,
                 'Vz': 5,
                 }


    ################################################################
    # train process
    ################################################################
    # grid = get_grid()
    # grid_real = get_grid()
    # grid = gen_uniform_grid(train_y[:1]).to(Device)
    post = cfdPost_2d()
    train_loader_sim = train_loader
    valid_loader_sim_1 = post.loader_similarity(valid_loader,
                                                grid=grids, scale=[-0.005, 0.005], expand=1, log=True,
                                                x_norm=x_normalizer_bc,
                                                y_norm=y_normalizer,
                                                inputdict=inputdict)
    valid_loader_sim_2 = post.loader_similarity(valid_loader, grid=grids, scale=[-0.01, 0.01], expand=1, log=True,
                                                x_norm=x_normalizer_bc,
                                                y_norm=y_normalizer,
                                                inputdict=inputdict)
    valid_loader_sim_3 = post.loader_similarity(valid_loader, grid=grids, scale=[-0.015, 0.015], expand=1, log=True,
                                                x_norm=x_normalizer_bc,
                                                y_norm=y_normalizer,
                                                inputdict=inputdict)
    del post

    for epoch in range(epochs):
        # if epoch<500:
        #     scale_value = (np.log10(1+(np.power(10,0.025)-1)*(epoch)/500))
        # else:
        #     scale_value = (np.log10(1 + (np.power(10, 0.025) - 1) * (1000-epoch) / 500))

        # post = cfdPost_2d()
        # train_loader_sim = post.loader_similarity(train_loader, grid=grids, scale=[-0.01, 0.01],
        #                                           x_norm=x_normalizer_bc,
        #                                           y_norm=y_normalizer,
        #                                           expand=1, log=True, inputdict=inputdict)
        # del post


        Net_model.train()
        log_loss[0].append(train(train_loader, Net_model, Device, Loss_func, Optimizer, Scheduler))

        Net_model.eval()
        log_loss[1].append(valid(valid_loader, Net_model, Device, Loss_func))
        log_loss[2].append(valid(valid_loader_sim_1, Net_model, Device, Loss_func))
        log_loss[3].append(valid(valid_loader_sim_2, Net_model, Device, Loss_func))
        log_loss[4].append(valid(valid_loader_sim_3, Net_model, Device, Loss_func))

        # print('epoch: {:6d}, lr: {:.3e}, train_step_loss: {:.3e}, valid_step_loss: {:.3e}, cost: {:.2f}'.
        #       format(epoch, learning_rate, log_loss[0][-1], log_loss[1][-1], time.time() - star_time))
        print('epoch: {:6d}, lr: {:.3e}, train_sim_step_loss: {:.3e}, valid_step_loss: {:.3e}, valid_sim_step_loss: {:.3e}, cost: {:.2f}'.
              format(epoch, learning_rate, log_loss[0][-1], log_loss[1][-1],  log_loss[2][-1], time.time() - star_time))

        star_time = time.time()

        if epoch > 0 and epoch % 10 == 0:
            fig, axs = plt.subplots(1, 1, figsize=(15, 8), num=1)
            Visual.plot_loss(fig, axs, np.arange(len(log_loss[0])), np.array(log_loss)[0, :], 'trainsim_step')
            Visual.plot_loss(fig, axs, np.arange(len(log_loss[0])), np.array(log_loss)[1, :], 'valid_step')
            Visual.plot_loss(fig, axs, np.arange(len(log_loss[0])), np.array(log_loss)[2, :], 'validsim1_step')
            Visual.plot_loss(fig, axs, np.arange(len(log_loss[0])), np.array(log_loss)[3, :], 'validsim2_step')
            Visual.plot_loss(fig, axs, np.arange(len(log_loss[0])), np.array(log_loss)[4, :], 'validsim3_step')

            fig.suptitle('training loss')
            fig.savefig(os.path.join(train_path, 'log_loss.svg'))
            plt.close(fig)

            torch.save({'log_loss': log_loss, 'net_model': Net_model.state_dict(), 'optimizer': Optimizer.state_dict()},
                    os.path.join(work_path, 'latest_model.pth'))

        if epoch==0:
            torch.save(Net_model,os.path.join(work_path, 'final_model.pth'))

            print(0)

        ################################################################
        # Visualization
        ################################################################
        if epoch % 100 == 0:
            train_source, train_true, train_pred = inference(train_loader, Net_model, Device)
            valid_source, valid_true, valid_pred = inference(valid_loader, Net_model, Device)



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



