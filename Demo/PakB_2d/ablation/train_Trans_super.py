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
from transformer.Transformers import FourierTransformer
from Tools.model_define.define_FNO import train, valid, inference, train_random, train_mask
from Demo.PakB_2d.ablation.ablation_PakB import get_loaders, get_setting, calculate_per
from Demo.PakB_2d.trains_PakB import train_supercondition, valid_supercondition, valid_detail, supredictor_list_windows
import wandb
os.chdir('E:\WQN\CODE\DENO4pytorch\Demo\PakB_2d/')


if __name__ == "__main__":
    ################################################################
    # configs
    ################################################################
    wandb_run = True
    patch_num = 1
    dataset_train_list = [
        # [1],
        [1, 2],
        [1, 2, 3],
        [1, 2, 3, 5],
        [1, 2, 3, 5, 10],
    ]
    basic_dict, train_dict, pred_model_dict, _ = get_setting()
    for save_number, dataset_train in enumerate(dataset_train_list):
        dataset_valid = [1, 2, 3, 5, 10, 15, 20]

        name = 'Trans'
        work_path = os.path.join('work_ablation', name + '_patch_' + str(patch_num) + '_super_' + str(save_number))
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
                name='Super_patch_'+str(patch_num)+'_' + str(dataset_train),
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
        Net_model = supredictor_list_windows(perd_model, super_model, channel_num=in_dim, win_split=patch_num).to(Device)
        # # 损失函数
        Loss_func_train = PakBAntiNormLoss(weighted_cof=0, shreshold_cof=-100, x_norm=x_normalizer, y_norm=y_normalizer)
        Loss_func_valid = PakBAntiNormLoss(weighted_cof=0, shreshold_cof=0, x_norm=x_normalizer, y_norm=y_normalizer)
        # # 优化算法
        Optimizer_0 = torch.optim.Adam(Net_model.pred_net.parameters(), lr=learning_rate, betas=(0.7, 0.9), weight_decay=1e-7)
        Optimizer_1 = torch.optim.Adam([
            {'params': Net_model.pred_net.parameters(), 'lr': learning_rate * 0.1},  # 学习率为默认的
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

        for hole_num in dataset_valid[:-2]:
            log_loss.update({'valid_loss_hole_' + str(hole_num): []})

        log_super_loss = {
            'train_super_loss': [],
        }
        for hole_num in dataset_valid:
            if hole_num <= in_dim:
                split_range = range(hole_num + 1)
            else:
                split_range = range(hole_num - in_dim, in_dim + 1)
            log_super_loss.update({'valid_super_loss_hole_' + str(hole_num): []})
            for split_num in split_range:
                log_super_loss.update({'valid_super_loss_hole_' + str(hole_num) + '_split_' + str(split_num) : []})
        ################################################################
        # train process
        ################################################################
        for epoch in range(epochs):

            Net_model.train()
            log_loss['train_loss'].append(
                train_supercondition(train_loader, Net_model, Device, Loss_func_train, Optimizer_0, Scheduler_0,
                                        x_norm=x_normalizer, super_num=0, channel_num=in_dim)
                                            )
            log_super_loss['train_super_loss'].append(
                train_supercondition(train_loader, Net_model, Device, Loss_func_train, Optimizer_1, Scheduler_1,
                                        x_norm=x_normalizer, super_num=1, channel_num=in_dim)
                                            )

            Net_model.eval()
            for hole_num, valid_loader_single in zip(dataset_valid[:-2], valid_loader_list[:-2]):
                log_loss['valid_loss_hole_' + str(hole_num)].append(
                    valid_supercondition(valid_loader_single, Net_model, Device, Loss_func_valid
                                         , x_norm=x_normalizer, super_num=0, channel_num=in_dim)
                )
            for hole_num, valid_loader_single in zip(dataset_valid, valid_loader_list):
                if hole_num <= in_dim:
                    split_range = range(hole_num + 1)
                else:
                    split_range = range(hole_num - in_dim, in_dim + 1)
                per_list = calculate_per(hole_num)
                total_sum = 0
                for split_num in split_range:
                    super_loss = valid_detail(valid_loader_single, Net_model, Device, Loss_func_valid,
                                     x_norm=x_normalizer,
                                     channel_num=in_dim,
                                     super_num=1, hole_num=hole_num, split_num=split_num,
                                     )
                    log_super_loss['valid_super_loss_hole_' + str(hole_num) + '_split_' + str(split_num)].append(super_loss)
                    total_sum = total_sum + super_loss * per_list[int(split_num)]
                log_super_loss['valid_super_loss_hole_' + str(hole_num)].append(total_sum)

            if epoch % 10 == 0:
                fig, axs = plt.subplots(1, 1, figsize=(15, 8), num=1)
                Visual.plot_loss(fig, axs, np.arange(len(log_loss['train_loss'])), np.array(log_loss['train_loss']),
                                 'train_step')
                fig.suptitle('training loss')
                fig.savefig(work.svg)
                plt.close(fig)
                torch.save(
                    {'log_loss': log_loss, 'net_model': Net_model.state_dict(), 'optimizer': Optimizer_0.state_dict()},
                    work.pth)
                torch.save(Net_model, work.fpth)
            print('epoch: {:6d}, '
                  'lr: {:.3e}, '
                  'train_step_loss: {:.3e}, '
                  'train_step_loss: {:.3e}, '
                  'cost: {:.2f}'.
                  format(epoch,
                         Optimizer_0.state_dict()['param_groups'][0]['lr'],
                         log_loss['train_loss'][-1],
                         log_super_loss['train_super_loss'][-1],
                         time.time() - star_time)
                  )
            star_time = time.time()
            if wandb_run:
                wandb.log({
                    **{key: value[-1] for key, value in log_loss.items()},
                    **{key: value[-1] for key, value in log_super_loss.items()},
                })
        if wandb_run: wandb.finish()







