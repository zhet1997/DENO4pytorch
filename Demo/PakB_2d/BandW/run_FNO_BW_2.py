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
from Utilizes.visual_data import MatplotlibVision, TextLogger
import matplotlib.pyplot as plt
import time
import yaml
from Demo.PakB_2d.utilizes_pakB import get_origin, PakBWeightLoss, get_loader_pakB, clear_value_in_hole
from Tools.train_model.model_whole_life import WorkPrj
from fno.FNOs import FNO2d
from Utilizes.loss_metrics import FieldsLpLoss
from Tools.model_define.define_FNO import train, valid, inference, train_random, train_mask
from Tools.pre_process.data_reform import data_padding, split_train_valid, get_loader_from_list
import wandb
os.chdir('E:\WQN\CODE\DENO4pytorch\Demo\PakB_2d/')

if __name__ == "__main__":
    ################################################################
    # configs
    ################################################################
    name = 'FNO'
    work_path = os.path.join('work', name + '_test_' + str(2))
    train_path = os.path.join(work_path)
    isCreated = os.path.exists(work_path)
    work = WorkPrj(work_path)
    Logger = TextLogger(os.path.join(train_path, 'train.log'))
    Device = work.device
    # data_para
    data_dict = {
    'in_dim' : 16,
    'out_dim' : 1,
    'ntrain' : 700,
    'nvalid' : 200,
    'dataset' : [1, 2, 3, 5, 10],
    }

    # train_para
    train_dict = {
    'batch_size' : 32,
    'epochs' : 801,
    'learning_rate' : 0.001,
    'scheduler_step' : 700,
    'scheduler_gamma' : 0.1,
    }
    # net_para
    Net_model_dict = {
    'modes' : (32, 32),
    'width' : 64,
    'depth' : 4,
    'steps' : 1,
    'padding' : 8,
    'dropout' : 0.1,
    }

    locals().update(data_dict)
    locals().update(train_dict)
    locals().update(Net_model_dict)

    wandb.init(
        # Set the project where this run will be logged
        project="pak_B_film_cooling_predictor",  # 写自己的
        entity="turbo-1997",
        notes="const=350",
        name='random_channel',
        # Track hyperparameters and run metadata
        config={
                **data_dict,
                **train_dict,
                **Net_model_dict,
                }
    )

    print(epochs, learning_rate, scheduler_step, scheduler_gamma)

    # ################################################################
    # # load data
    # ################################################################
    train_input_list = []
    train_output_list = []
    valid_input_list = []
    valid_output_list = []

    for kk, hole_num in enumerate(dataset):
        design, fields, grids = get_origin(type='struct', hole_num=hole_num, realpath=os.path.join('data'))  # 获取原始数据取原始数据
        input = data_padding(design, const=350, channel_num=16)
        output = fields
        # print(input.shape, output.shape)
        train_i, valid_i = split_train_valid(input, train_num=ntrain, valid_num=nvalid)
        train_o, valid_o = split_train_valid(output, train_num=ntrain, valid_num=nvalid)

        train_input_list.append(train_i.copy())
        train_output_list.append(train_o.copy())
        valid_input_list.append(valid_i.copy())
        valid_output_list.append(valid_o.copy())

    train_loader, x_normalizer, y_normalizer = get_loader_from_list(train_input_list,
                                                                   train_output_list,
                                                                   batch_size=batch_size,
                                                                   )
    valid_loader, _, _ = get_loader_from_list(valid_input_list,
                                                   valid_output_list,
                                                   x_normalizer=x_normalizer,
                                                   y_normalizer=y_normalizer,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   combine_list=True,
                                                   )
    # valid_loader_list, _, _ = get_loader_from_list(valid_input_list,
    #                                          valid_output_list,
    #                                          x_normalizer=x_normalizer,
    #                                          y_normalizer=y_normalizer,
    #                                          batch_size=batch_size,
    #                                          shuffle=True,
    #                                          combine_list=False,
    #                                          )
    # ################################################################
    # #  Neural Networks
    # ################################################################
    #
    # # 建立网络
    Net_model = FNO2d(in_dim=in_dim, out_dim=out_dim, modes=modes, width=width, depth=depth, steps=steps,
                      padding=padding, activation='gelu').to(Device)
    # # 损失函数
    Loss_func = nn.MSELoss()
    Loss_func_valid = PakBWeightLoss(weighted_cof=0, shreshold_cof=0, x_norm=x_normalizer)
    Loss_real = FieldsLpLoss(p=2, d=2)
    # # 优化算法
    Optimizer = torch.optim.Adam(Net_model.parameters(), lr=learning_rate, betas=(0.7, 0.9),)# weight_decay=1e-7)
    # # 下降策略
    Scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    # # 可视化
    Visual = MatplotlibVision(work_path, input_name=('x', 'y'), field_name=('T',))

    star_time = time.time()
    log_loss = {
        'train_step_loss': [],
        'valid_step_loss': [],
    }
    ################################################################
    # load data
    ################################################################
    isExist = os.path.exists(work.pth)
    if isExist:
        checkpoint = torch.load(work.pth, map_location=Device)
        Net_model.load_state_dict(checkpoint['net_model'])
        Net_model.eval()
    ################################################################
    # train process
    ################################################################
    for epoch in range(epochs):

        Net_model.train()
        log_loss['train_step_loss'].append(train_random(train_loader, Net_model, Device, Loss_func, Optimizer, Scheduler))

        Net_model.eval()
        log_loss['valid_step_loss'].append(valid(valid_loader, Net_model, Device, Loss_func_valid))

        # if epoch > 0 and epoch % 10 == 0:
        if epoch % 10 == 0:
            fig, axs = plt.subplots(1, 1, figsize=(15, 8), num=1)
            Visual.plot_loss(fig, axs, np.arange(len(log_loss['train_step_loss'])), np.array(log_loss['train_step_loss']), 'train_step')
            Visual.plot_loss(fig, axs, np.arange(len(log_loss['valid_step_loss'])), np.array(log_loss['valid_step_loss']), 'valid_step')
            fig.suptitle('training loss')
            fig.savefig(os.path.join(train_path, 'log_loss.svg'))
            plt.close(fig)

        ################################################################
        # Visualization
        ################################################################
        if epoch % 10 == 0:
            train_source, train_true, train_pred = inference(train_loader, Net_model, Device)
            train_pred = clear_value_in_hole(train_pred, train_source, x_norm=x_normalizer)
            train_true = clear_value_in_hole(train_true, train_source, x_norm=x_normalizer)
            train_pred = y_normalizer.back(train_pred)
            train_true = y_normalizer.back(train_true)

            # for valid_loader in valid_loader_list:
            valid_source, valid_true, valid_pred = inference(valid_loader, Net_model, Device)
            valid_pred = clear_value_in_hole(valid_pred, valid_source, x_norm=x_normalizer)
            valid_true = clear_value_in_hole(valid_true, valid_source, x_norm=x_normalizer)
            valid_pred = y_normalizer.back(valid_pred)
            valid_true = y_normalizer.back(valid_true)

            train_abs_loss = Loss_real.abs(train_true, train_pred)
            train_rel_loss = Loss_real.rel(train_true, train_pred)
            valid_abs_loss = Loss_real.abs(valid_true, valid_pred)
            valid_rel_loss = Loss_real.rel(valid_true, valid_pred)

        if epoch > 0 and epoch % 100 == 0:
            for fig_id in range(15):
                fig, axs = plt.subplots(out_dim, 3, figsize=(18, 6), num=2)
                Visual.plot_fields_ms(fig, axs, train_true[fig_id], train_pred[fig_id], grids)
                fig.savefig(os.path.join(work_path, 'train_solution_' + str(fig_id) + '.jpg'))
                plt.close(fig)

            for fig_id in range(15):
                fig, axs = plt.subplots(out_dim, 3, figsize=(18, 6), num=3)
                Visual.plot_fields_ms(fig, axs, valid_true[fig_id], valid_pred[fig_id], grids)
                fig.savefig(os.path.join(work_path, 'valid_solution_' + str(fig_id) + '.jpg'))
                plt.close(fig)

            torch.save({'log_loss': log_loss, 'net_model': Net_model.state_dict(), 'optimizer': Optimizer.state_dict()},work.pth)
            torch.save(Net_model, work.fpth)

        print('epoch: {:6d}, '
              'lr: {:.3e}, '
              'train_step_loss: {:.3e}, '
              'valid_step_loss: {:.3e}, '
              'cost: {:.2f}'.
              format(epoch,
                     Optimizer.state_dict()['param_groups'][0]['lr'],
                     log_loss['train_step_loss'][-1],
                     log_loss['valid_step_loss'][-1],
                     time.time() - star_time)
              )

        star_time = time.time()
        wandb.log({
            "train_step_loss": log_loss['train_step_loss'][-1],
            "valid_step_loss": log_loss['valid_step_loss'][-1],
            'train_abs_loss': float(np.mean(train_abs_loss)),
            'train_rel_loss': float(np.mean(train_rel_loss)),
            'valid_abs_loss': float(np.mean(valid_abs_loss)),
            'valid_rel_loss': float(np.mean(valid_rel_loss)),
        })

    wandb.finish()






