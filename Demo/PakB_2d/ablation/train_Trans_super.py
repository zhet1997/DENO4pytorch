import os
import numpy as np
import torch
import torch.nn as nn
from Utilizes.visual_data import MatplotlibVision, TextLogger
import matplotlib.pyplot as plt
import time
import yaml
from Demo.PakB_2d.utilizes_pakB import get_origin, PakBAntiNormLoss, get_loader_pakB, clear_value_in_hole
from Tools.train_model.model_whole_life import WorkPrj
from fno.FNOs import FNO2d
from transformer.Transformers import FourierTransformern
from Utilizes.loss_metrics import FieldsLpLoss
from Tools.model_define.define_FNO import train, valid, inference, train_random, train_mask
from Tools.pre_process.data_reform import data_padding, split_train_valid, get_loader_from_list, get_loader_from_list_combine
from Demo.PakB_2d.trains_PakB import train_supercondition, valid_supercondition, valid_detail, supredictor_list_windows
import wandb
os.chdir('E:\WQN\CODE\DENO4pytorch\Demo\PakB_2d/')


if __name__ == "__main__":
    ################################################################
    # configs
    ################################################################
    dataset_train = [1, 2, 3, 5, 10]
    dataset_valid = [1, 2, 3, 5, 10]
    wandb_run = True
    name = 'Trans'
    work_path = os.path.join('work_ablation', name + '_baseline_' + str(save_number))
    work = WorkPrj(work_path)
    Logger = TextLogger(os.path.join(work_path, 'train.log'))
    Device = work.device

    basic_dict, train_dict, pred_model_dict, super_model_dict = get_setting()
    locals().update(basic_dict)
    locals().update(train_dict)

    basic_dict['work_path'] = work_path
    channel_num = in_dim

    if wandb_run:
        wandb.init(
            project="pak_B_film_cooling_ablation_500",  # 写自己的
            entity="turbo-1997",
            notes="const=350",
            name='Baseline_super_' + str(dataset_train),
            config={
                **basic_dict,
                **train_dict,
                **pred_model_dict,
                **super_model_dict,
            }
        )

    print(epochs, learning_rate, scheduler_step, scheduler_gamma)

    # ################################################################
    # # load data
    # ################################################################
    train_loader, valid_loader_list, x_normalizer, y_normalizer = get_loaders(dataset_train,
                                                                              dataset_valid,
                                                                              train_num=ntrain,
                                                                              valid_num=nvalid,
                                                                              channel_num=channel_num,
                                                                              batch_size=batch_size,
                                                                              )
    # ################################################################
    # #  Neural Networks
    # ################################################################
    #
    # # 建立网络
    perd_model = FourierTransformer(**pred_model_dict).to(Device)
    super_model = FNO2d(in_dim=2, out_dim=1, **super_model_dict).to(Device)
    Net_model = supredictor_list_windows(perd_model, super_model, channel_num=in_dim).to(Device)
    # # 损失函数
    Loss_func_train = PakBAntiNormLoss(weighted_cof=0, shreshold_cof=-10, x_norm=x_normalizer, y_norm=y_normalizer)
    Loss_func_valid = PakBAntiNormLoss(weighted_cof=0, shreshold_cof=0, x_norm=x_normalizer, y_norm=y_normalizer)
    # # 优化算法
    Optimizer = torch.optim.Adam(Net_model.parameters(), lr=learning_rate, betas=(0.7, 0.9), )  # weight_decay=1e-7)
    Scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    Visual = MatplotlibVision(work_path, input_name=('x', 'y'), field_name=('T',))

    # # 优化算法
    Optimizer_0 = torch.optim.Adam(pred_net_parameters.values(), lr=learning_rate, betas=(0.7, 0.9), weight_decay=1e-7)
    Optimizer_1 = torch.optim.Adam([
        {'params': Net_model.pred_net.parameters(), 'lr': learning_rate / 10},  # 学习率为默认的
        {'params': Net_model.super_net.parameters()}], lr=learning_rate, betas=(0.7, 0.9), weight_decay=1e-7)
    # # 下降策略
    Scheduler_0 = torch.optim.lr_scheduler.StepLR(Optimizer_0, step_size=scheduler_step, gamma=scheduler_gamma)
    Scheduler_1 = torch.optim.lr_scheduler.StepLR(Optimizer_1, step_size=scheduler_step, gamma=scheduler_gamma)
    # # 可视化
    Visual = MatplotlibVision(work_path, input_name=('x', 'y'), field_name=('T',))

    star_time = time.time()
    log_loss = {
        'train_loss': [],
    }

    log_super_loss = {
        'train_super_loss': [],
    }
    for hole_num in dataset_valid:
        if hole_num <= in_dim:
            split_range = range(hole_num + 1)
        else:
            split_range = range(hole_num - in_dim, in_dim + 1)
        for split_num in split_range:
            log_super_loss.update({'valid_super_loss_hole_' + str(hole_num) + '_split_' + str(split_num) : []})
    ################################################################
    # train process
    ################################################################
    for epoch in range(epochs):

        Net_model.train()
        log_loss['train_step_loss'].append(
            train(train_loader, Net_model, Device, Loss_func, Optimizer, Scheduler))
        log_loss['train_step_loss_1'].append(
            train(train_loader_1, Net_model, Device, Loss_func, Optimizer_2, Scheduler_2))

        Net_model.eval()
        log_loss['valid_step_loss'].append(valid(valid_loader_0, Net_model, Device, Loss_func_valid))
        log_loss['valid_step_loss_1'].append(valid(valid_loader_1, Net_model, Device, Loss_func_valid))

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
            train_source, train_true, train_pred = inference(train_loader_0, Net_model, Device)
            train_pred = clear_value_in_hole(train_pred, train_source, x_norm=x_normalizer)
            train_true = clear_value_in_hole(train_true, train_source, x_norm=x_normalizer)
            train_pred = y_normalizer.back(train_pred)
            train_true = y_normalizer.back(train_true)

            # for valid_loader in valid_loader_list:
            valid_source, valid_true, valid_pred = inference(valid_loader_0, Net_model, Device)
            valid_pred = clear_value_in_hole(valid_pred, valid_source, x_norm=x_normalizer)
            valid_true = clear_value_in_hole(valid_true, valid_source, x_norm=x_normalizer)
            valid_pred = y_normalizer.back(valid_pred)
            valid_true = y_normalizer.back(valid_true)

            train_abs_loss = Loss_real.abs(train_true, train_pred)
            train_rel_loss = Loss_real.rel(train_true, train_pred)
            valid_abs_loss = Loss_real.abs(valid_true, valid_pred)
            valid_rel_loss = Loss_real.rel(valid_true, valid_pred)

        if epoch % 10 == 0:
            for hole_num, valid_loader_single in zip(dataset, valid_loader_list):
                if hole_num <= in_dim:
                    split_range = range(hole_num+1)
                else:
                    split_range = range(hole_num - in_dim, in_dim + 1)
                for split_num in split_range:
                    log_loss_detail['valid_deatil_loss_1_hole_'+str(hole_num)+'_split_'+str(split_num)].append(
                        valid_detail(valid_loader_single, Net_model, Device, Loss_func_valid,
                                     x_norm=x_normalizer,
                                     channel_num=in_dim,
                                     super_num=1, hole_num=hole_num, split_num=split_num,
                                     ))

        if epoch > 0 and epoch % 100 == 0:
            for fig_id in range(5):
                fig, axs = plt.subplots(out_dim, 3, figsize=(18, 6), num=2)
                Visual.plot_fields_ms(fig, axs, train_true[fig_id], train_pred[fig_id], grids)
                fig.savefig(os.path.join(work_path, 'train_solution_' + str(fig_id) + '.jpg'))
                plt.close(fig)

            for fig_id in range(5):
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
            "train_step_loss_1": log_loss['train_step_loss_1'][-1],
            "valid_step_loss_1": log_loss['valid_step_loss_1'][-1],
            'train_abs_loss': float(np.mean(train_abs_loss)),
            'train_rel_loss': float(np.mean(train_rel_loss)),
            'valid_abs_loss': float(np.mean(valid_abs_loss)),
            'valid_rel_loss': float(np.mean(valid_rel_loss)),
            **{key: value[-1] for key, value in log_loss_detail.items()},
        })
    wandb.finish()






