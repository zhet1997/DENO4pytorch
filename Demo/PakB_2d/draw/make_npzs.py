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

os.chdir(r'E:\WQN\CODE\DENO4pytorch')
def train(dataloader, netmodel, device, lossfunc, optimizer, scheduler):
    """
    Args:
        data_loader: output fields at last time step
        netmodel: Network
        lossfunc: Loss function
        optimizer: optimizer
        scheduler: scheduler
    """
    grid = gen_uniform_grid(torch.tensor(np.zeros([1, 128, 128, 8]))).to(device)
    train_loss = 0
    for batch, (xx, yy) in enumerate(dataloader):
        for ii in range(xx.shape[0]):
            idx = torch.randperm(xx.shape[-1])
            xx[ii] = xx[ii,..., idx]
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


def train_mask(dataloader, netmodel, device, lossfunc, optimizer, scheduler, x_norm=None):
    """
    Args:
        data_loader: output fields at last time step
        netmodel: Network
        lossfunc: Loss function
        optimizer: optimizer
        scheduler: scheduler
    """
    grid = gen_uniform_grid(torch.tensor(np.zeros([1, 128, 128, 8]))).to(device)
    train_loss = 0
    for batch, (xx, yy) in enumerate(dataloader):
        xx = xx.to(device)
        yy = yy.to(device)

        xx_mask = torch.zeros([*xx.shape[:-1],1], device=device)
        xx_none = torch.zeros([*xx.shape[1:-1],1], device=device) + 350
        xx_none = x_norm.norm(xx_none)
        for ii in range(xx.shape[0]):
            idx = torch.randperm(xx.shape[-1])
            xx[ii] = xx[ii,..., idx]
            xx_mask[ii] = xx[ii, ..., idx[0]:idx[0]+1]
            xx[ii, ..., idx[0]:idx[0]+1] = xx_none.clone().detach()

        coords = grid.tile([xx.shape[0], 1, 1, 1])

        pred = netmodel(xx, coords)
        loss = lossfunc(pred, yy, xx_mask)

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
    grid = gen_uniform_grid(torch.tensor(np.zeros([1, 128, 128, 8]))).to(device)
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
    grid = gen_uniform_grid(torch.tensor(np.zeros([1, 128, 128, 8]))).to(device)
    with torch.no_grad():
        xx, yy = next(iter(dataloader))
        xx = xx.to(device)
        coords = grid.tile([xx.shape[0], 1, 1, 1])
        pred = netmodel(xx, coords)

    # equation = model.equation(u_var, y_var, out_pred)
    return xx.cpu().numpy(), yy.numpy(), pred.cpu().numpy()

def get_loader_pakB(
            train_x, train_y,
            x_normalizer=None,
            y_normalizer=None,
           ):
    if x_normalizer is None:
        x_normalizer = DataNormer(train_x, method='mean-std', axis=(0,1,2,3))
    if y_normalizer is None:
        y_normalizer = DataNormer(train_y, method='mean-std')

    train_x = x_normalizer.norm(train_x)
    train_y = y_normalizer.norm(train_y)

    train_x = torch.as_tensor(train_x, dtype=torch.float)
    train_y = torch.as_tensor(train_y, dtype=torch.float)

    #
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y),
                                               batch_size=batch_size, shuffle=True, drop_last=True)

    return train_loader, x_normalizer, y_normalizer

if __name__ == "__main__":
    ################################################################
    # configs
    ################################################################
    name = 'Trans'
    work_path = os.path.join('Demo', 'PakB_2d', 'work', name+'_'+str(9) + '_mask_1+2')
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
    scheduler_step = 100
    scheduler_gamma = 0.5
    r1 = 1
    print(epochs, learning_rate, scheduler_step, scheduler_gamma)
    # #这部分应该是重采样
    # #不进行稀疏采样
    # ################################################################
    # # load data
    # ################################################################
    work = WorkPrj(work_path)
    x_normalizer = None
    train_input_list = []
    train_output_list = []
    valid_input_list = []
    valid_output_list = []
    valid_expand_input_list = []
    valid_expand_output_list = []
    for kk, hole_num in enumerate([3, 5, 10]):
        design, fields, grids = get_origin(type='struct', hole_num=hole_num)  # 获取原始数据取原始数据
        input = np.zeros([*design.shape[:-1], 16]) + 350
        input[..., :hole_num] = design.copy()
        output = fields
        print(input.shape, output.shape)

        train_input_list.append(input[ntrain:].copy())
        train_output_list.append(output[ntrain:].copy())

        valid_input_list.append(input[:-nvalid].copy())
        valid_output_list.append(output[:-nvalid].copy())

    for kk, hole_num in enumerate([1, 2]):
        design, fields, grids = get_origin(type='struct', hole_num=hole_num)  # 获取原始数据取原始数据
        input = np.zeros([*design.shape[:-1], 16]) + 350
        input[..., :hole_num] = design.copy()
        output = fields
        print(input.shape, output.shape)

        # train_input_list.append(input[ntrain:].copy())
        # train_output_list.append(output[ntrain:].copy())

        valid_expand_input_list.append(input[:-nvalid].copy())
        valid_expand_output_list.append(output[:-nvalid].copy())

    train_loader, x_normalizer, y_normalizer = get_loader_pakB(np.concatenate(train_input_list,
                                                                            axis=0),
                                                             np.concatenate(train_output_list,
                                                                            axis=0),
                                                             )
    valid_loader, _, _ = get_loader_pakB(np.concatenate(valid_input_list,
                                                        axis=0),
                                         np.concatenate(valid_output_list,
                                                        axis=0),
                                         x_normalizer=x_normalizer,
                                         y_normalizer=y_normalizer,
                                         )

    valid_expand_loader, _, _ = get_loader_pakB(np.concatenate(valid_expand_input_list,
                                                        axis=0),
                                         np.concatenate(valid_expand_output_list,
                                                        axis=0),
                                         x_normalizer=x_normalizer,
                                         y_normalizer=y_normalizer,
                                         )

    #
    #
    # ################################################################
    # #  Neural Networks
    # ################################################################
    with open(os.path.join('Demo', 'PakB_2d', 'data', 'configs', 'transformer_config_pakb.yml')) as f:
        config = yaml.full_load(f)
        config = config['PakB_2d']
    #
    # # 建立网络
    Net_model = FourierTransformer(**config).to(Device)

    # Tra_model = FourierTransformer(**config).to(Device)
    # # Tra_model = SimpleTransformer(**config).to(Device)
    # # FNO_model = FNO2d(in_dim=2, out_dim=config['n_targets'], modes=(16, 16), width=64, depth=4,
    # #                   padding=9, activation='gelu').to(Device)
    # MLP_model = FcnSingle(planes=(in_dim, 64, 64, 64, config['n_targets']), last_activation=True).to(Device)
    # # Net_model = predictor(trunc=Tra_model, branch=MLP_model, field_dim=out_dim).to(Device)
    # Share_model = FcnSingle(planes=(config['n_targets']+1, 64, 64, out_dim), last_activation=False).to(Device)
    #
    # super_model = FcnSingle(planes=(config['n_targets']*2, 64, 64, config['n_targets']), last_activation=False).to(Device)
    # Net_model = predictor(trunc=Tra_model, branch=MLP_model, supercondtion=super_model, share=Share_model, super_order=[0,]).to(Device)
    log_loss = [[], [], []]
    isExist = os.path.exists(work.pth)

    if isExist:
        checkpoint = torch.load(work.pth, map_location=Device)
        Net_model.load_state_dict(checkpoint['net_model'])
        log_loss = checkpoint['log_loss']
        #


    # model_statistics = summary(Net_model, input_size=(batch_size, train_x.shape[1]), device=str(Device))
    # Logger.write(str(model_statistics))
    #
    # # 损失函数
    Loss_func = nn.MSELoss()
    Loss_func_mask = PakBWeightLoss(weighted_cof=0, shreshold_cof=50, x_norm=x_normalizer)
    # # Loss_func = nn.SmoothL1Loss()
    # # 优化算法
    Optimizer = torch.optim.Adam(Net_model.parameters(), lr=learning_rate, betas=(0.7, 0.9),)# weight_decay=1e-7)
    # # 下降策略
    Scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    # # 可视化
    Visual = MatplotlibVision(work_path, input_name=('x', 'y'), field_name=('TEMPERATURE',))


    star_time = time.time()


    ################################################################
    # train process
    ################################################################
    # grid = get_grid()
    # grid_real = get_grid()
    # grid = gen_uniform_grid(train_y[:1]).to(Device)
    Net_model.eval()


    train_source, train_true, train_pred = inference(train_loader, Net_model, Device)
    valid_source, valid_true, valid_pred = inference(valid_loader, Net_model, Device)

    train_pred = y_normalizer.back(train_pred)
    train_true = y_normalizer.back(train_true)
    valid_pred = y_normalizer.back(valid_pred)
    valid_true = y_normalizer.back(valid_true)

    for fig_id in range(32):
        fig, axs = plt.subplots(out_dim, 3, figsize=(18, 6), num=2)
        Visual.plot_fields_ms(fig, axs, train_true[fig_id], train_pred[fig_id], grids)
        fig.savefig(os.path.join(work_path, 'train_solution_' + str(fig_id) + '.jpg'))
        plt.close(fig)

    for fig_id in range(32):
        fig, axs = plt.subplots(out_dim, 3, figsize=(18, 6), num=3)
        Visual.plot_fields_ms(fig, axs, valid_true[fig_id], valid_pred[fig_id], grids)
        fig.savefig(os.path.join(work_path, 'valid_solution_' + str(fig_id) + '.jpg'))
        plt.close(fig)

    save_dict = {}
    save_dict.update({'x': x})
    save_dict.update({'true': true})
    save_dict.update({'pred': pred})
    np.savez(path_save, **save_dict)



