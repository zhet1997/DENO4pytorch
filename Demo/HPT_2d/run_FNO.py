
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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from torchinfo import summary
# from torchsummary import summary
from fno.FNOs import FNO2d
from Utilizes.visual_data import MatplotlibVision
from Utilizes.process_data import DataNormer
import matplotlib.pyplot as plt
import time
from Demo.HPT_2d.utilizes_HPT import get_origin
from Tools.post_process.post_CFD import cfdPost_2d
from Demo.GVRB_2d.train_model_GVRB.model_whole_life import WorkPrj
from Utilizes.geometrics import gen_uniform_grid


def train(dataloader, netmodel, device, lossfunc, optimizer, scheduler):
    """
    Args:
        data_loader: output fields at last time step
        netmodel: Network
        lossfunc: Loss function
        optimizer: optimizer
        scheduler: scheduler
    """
    grid = gen_uniform_grid(torch.tensor(np.zeros([1, 64, 128, 8]))).to(device)
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
    grid = gen_uniform_grid(torch.tensor(np.zeros([1, 64, 128, 8]))).to(device)
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
    grid = gen_uniform_grid(torch.tensor(np.zeros([1, 64, 128, 8]))).to(device)
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
    name = 'FNO_'
    work_path = os.path.join('work', name + '_batch_num_' + str(2.3))
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

    in_dim = 5
    out_dim = 8

    batch_size = 8
    batch_number = 2

    ntrain = batch_number * batch_size
    nvalid = 640

    epochs = 501
    learning_rate = 0.001
    scheduler_step = 300
    scheduler_gamma = 0.1

    # net setting
    modes = (8, 8)
    width = 64
    depth = 6
    steps = 1
    padding = 8
    dropout = 0.0

    r1 = 1

    boundaryDict = {'Flow Angle': 0,
                    'Absolute Total Temperature': 1,
                    'Absolute Total Pressure': 2,
                    'Rotational_speed': 3,
                    'Shroud_Gap': 4
                    }


    print(epochs, learning_rate, scheduler_step, scheduler_gamma)
    # #这部分应该是重采样
    # #不进行稀疏采样
    #
    #
    # ################################################################
    # # load data
    # ################################################################

    design, fields, grids = get_origin(type='struct',
                                       realpath='E:\WQN\CODE\DENO4pytorch\Demo\HPT_2d\data')  # 获取原始数据取原始数据
    work = WorkPrj(work_path)
    input = design
    input = np.tile(design[:, None, None, :], (1, 64, 128, 1))
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
    x_normalizer = DataNormer(train_x.numpy(), method='mean-std')
    x_normalizer.dim_change(2)

    # x_normalizer_bc = DataNormer(train_x.numpy(), method='mean-std')
    # x_normalizer_bc.dim_change(2)
    # x_normalizer_bc.shrink(slice(96, 100, 1))

    y_normalizer = DataNormer(train_y.numpy(), method='mean-std')
    y_normalizer.dim_change(2)

    #########################################################################################################
    # self-supervise data genration
    virtual_batchs = batch_number
    ##########################################################################################################
    # x_normalizer = DataNormer(train_x.numpy(), method='mean-std')
    train_x = x_normalizer.norm(train_x)
    valid_x = x_normalizer.norm(valid_x)

    # y_normalizer = DataNormer(train_y.numpy(), method='mean-std')
    train_y = y_normalizer.norm(train_y)
    valid_y = y_normalizer.norm(valid_y)
    #
    x_normalizer.save(os.path.join(work_path, 'x_norm.pkl'))  # 将normalizer保存下来
    y_normalizer.save(os.path.join(work_path, 'y_norm.pkl'))
    #
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
    # 优化算法
    Optimizer = torch.optim.AdamW(Net_model.parameters(), lr=learning_rate, betas=(0.7, 0.9), weight_decay=1e-7)
    # 下降策略
    Scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    # 可视化
    Visual = MatplotlibVision(work_path, input_name=('x', 'y'), field_name=('ps', 'ts', 'rho', 'vx', 'vy', 'vz', 'tt1', 'tt2'))

    star_time = time.time()
    log_loss = [[], [], []]

    ################################################################
    # train process
    ################################################################

    post = cfdPost_2d()
    valid_loader_sim = post.loader_similarity(valid_loader,
                                              boundarydict=boundaryDict,
                                              grid=grids, scale=[-0.03, 0.03], expand=1, log=True,
                                              x_norm=x_normalizer,
                                              y_norm=y_normalizer,
                                              )

    for epoch in range(epochs):

        Net_model.train()
        log_loss[0].append(train(train_loader, Net_model, Device, Loss_func, Optimizer, Scheduler))

        Net_model.eval()
        log_loss[1].append(valid(valid_loader, Net_model, Device, Loss_func))
        log_loss[2].append(valid(valid_loader_sim, Net_model, Device, Loss_func))
        print('epoch: {:6d}, lr: {:.3e}, train_step_loss: {:.3e}, valid_step_loss: {:.3e}, cost: {:.2f}'.
              format(epoch, Optimizer.param_groups[0]['lr'], log_loss[0][-1], log_loss[1][-1], time.time() - star_time))

        star_time = time.time()

        if epoch > 0 and epoch % 10 == 0:
            fig, axs = plt.subplots(1, 1, figsize=(15, 8), num=1)
            Visual.plot_loss(fig, axs, np.arange(len(log_loss[0])), np.array(log_loss)[0, :], 'train_step')
            Visual.plot_loss(fig, axs, np.arange(len(log_loss[0])), np.array(log_loss)[1, :], 'valid_step')
            Visual.plot_loss(fig, axs, np.arange(len(log_loss[0])), np.array(log_loss)[2, :], 'valid_sim_step')
            fig.suptitle('training loss')
            fig.savefig(os.path.join(work_path, 'log_loss.svg'))
            plt.close(fig)

        ################################################################
        # Visualization
        ################################################################

        if epoch > 0 and epoch % 100 == 0:
            # print('epoch: {:6d}, lr: {:.3e}, eqs_loss: {:.3e}, bcs_loss: {:.3e}, cost: {:.2f}'.
            #       format(epoch, learning_rate, log_loss[-1][0], log_loss[-1][1], time.time()-star_time))
            train_source, train_true, train_pred = inference(train_loader, Net_model, Device)
            valid_source, valid_true, valid_pred = inference(valid_loader, Net_model, Device)

            torch.save({'log_loss': log_loss, 'net_model': Net_model.state_dict(), 'optimizer': Optimizer.state_dict()},
                       os.path.join(work_path, 'latest_model.pth'))

            # for fig_id in range(5):
            #     fig, axs = plt.subplots(out_dim, 3, figsize=(18, 20), num=2)
            #     Visual.plot_fields_ms(fig, axs, train_true[fig_id], train_pred[fig_id], grid)
            #     fig.savefig(os.path.join(work_path, 'train_solution_' + str(fig_id) + '.jpg'))
            #     plt.close(fig)
            #
            # for fig_id in range(5):
            #     fig, axs = plt.subplots(out_dim, 3, figsize=(18, 20), num=3)
            #     Visual.plot_fields_ms(fig, axs, valid_true[fig_id], valid_pred[fig_id], grid)
            #     fig.savefig(os.path.join(work_path, 'valid_solution_' + str(fig_id) + '.jpg'))
            #     plt.close(fig)

