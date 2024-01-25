import os
import numpy as np
import torch
import torch.nn as nn
from Utilizes.visual_data import MatplotlibVision, TextLogger
import matplotlib.pyplot as plt
import time
import yaml
from Demo.PakB_2d.utilizes_pakB import get_origin, PakBAntiNormLoss, clear_value_in_hole
from Tools.train_model.model_whole_life import WorkPrj
from transformer.Transformers import FourierTransformer
from Tools.model_define.define_FNO import train, valid, inference, train_random, train_mask
from Demo.PakB_2d.ablation.ablation_PakB import get_loaders, get_setting
import wandb
os.chdir('E:\WQN\CODE\DENO4pytorch\Demo\PakB_2d/')

# def baseline_train(dataset_train=[1, 2, 3, 5, 10],
#                    dataset_valid=[1, 2, 3, 5, 10],
#                    wandb_run=True,
#                    save_number=None,
#                    ):
if __name__ == "__main__":
    dataset_train_list = [
        [1],
        [1,2],
        [1,2,3],
        [1,2,3,5],
        [1,2,3,5,10],
    ]
    basic_dict, train_dict, pred_model_dict, _ = get_setting()
    for save_number, dataset_train in enumerate(dataset_train_list):
        dataset_valid = [1, 2, 3, 5, 10]
        wandb_run = True
        name = 'Trans'
        work_path = os.path.join('work_ablation', name + '_baseline_' + str(save_number))
        work = WorkPrj(work_path)
        Logger = TextLogger(os.path.join(work_path, 'train.log'))
        Device = work.device

        # basic_dict, train_dict, pred_model_dict = get_setting()
        basic_dict['work_path'] = work_path

        locals().update(basic_dict)
        locals().update(train_dict)
        # locals().update(pred_model_dict)
        channel_num = in_dim

        if wandb_run:
            wandb.init(
                project="pak_B_film_cooling_ablation_500",  # 写自己的
                entity="turbo-1997",
                notes="const=350",
                name='Baseline_random_' + str(dataset_train),
                config={
                    **basic_dict,
                    **train_dict,
                    **pred_model_dict,
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
        # # 建立网络
        Net_model = FourierTransformer(**pred_model_dict).to(Device)
        # # 损失函数
        Loss_func_train = PakBAntiNormLoss(weighted_cof=0, shreshold_cof=-10, x_norm=x_normalizer, y_norm=y_normalizer)
        Loss_func_valid = PakBAntiNormLoss(weighted_cof=0, shreshold_cof=0, x_norm=x_normalizer, y_norm=y_normalizer)
        # # 优化算法
        Optimizer = torch.optim.Adam(Net_model.parameters(), lr=learning_rate, betas=(0.7, 0.9), )  # weight_decay=1e-7)
        Scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
        Visual = MatplotlibVision(work_path, input_name=('x', 'y'), field_name=('T',))

        star_time = time.time()
        log_loss = {
            'train_loss': [],
        }
        for hole_num in dataset_valid:
            log_loss.update({'valid_loss_hole_' + str(hole_num): []})
        ################################################################
        # train process
        ################################################################
        for epoch in range(epochs):

            Net_model.train()
            log_loss['train_loss'].append(train(train_loader, Net_model, Device, Loss_func_train, Optimizer, Scheduler))
            Net_model.eval()
            for hole_num, valid_loader_single in zip(dataset_valid, valid_loader_list):
                log_loss['valid_loss_hole_' + str(hole_num)].append(
                    valid(valid_loader_single, Net_model, Device, Loss_func_valid)
                )
            if epoch % 10 == 0:
                fig, axs = plt.subplots(1, 1, figsize=(15, 8), num=1)
                Visual.plot_loss(fig, axs, np.arange(len(log_loss['train_loss'])), np.array(log_loss['train_loss']),
                                 'train_step')
                fig.suptitle('training loss')
                fig.savefig(work.svg)
                plt.close(fig)
                torch.save(
                    {'log_loss': log_loss, 'net_model': Net_model.state_dict(), 'optimizer': Optimizer.state_dict()},
                    work.pth)
                torch.save(Net_model, work.fpth)
            print('epoch: {:6d}, '
                  'lr: {:.3e}, '
                  'train_step_loss: {:.3e}, '
                  'valid_loss_hole_1: {:.3e}, '
                  'cost: {:.2f}'.
                  format(epoch,
                         Optimizer.state_dict()['param_groups'][0]['lr'],
                         log_loss['train_loss'][-1],
                         log_loss['valid_loss_hole_1'][-1],
                         time.time() - star_time)
                  )
            star_time = time.time()
            if wandb_run:
                wandb.log({
                    **{key: value[-1] for key, value in log_loss.items()}
                })
        if wandb_run: wandb.finish()







