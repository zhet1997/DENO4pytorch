import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import argparse

sys.path.append('/data/wqn/DENO4pytorch')
sys.path.append('/data/wqn/DENO4pytorch/Models')
sys.path.append('/data/wqn/DENO4pytorch/Utilizes')

from Utilizes.visual_data import MatplotlibVision, TextLogger
import matplotlib.pyplot as plt
from Tools.train_model.model_whole_life import WorkPrj
from fno.FNOs import FNO2d
from transformer.Transformers import FourierTransformer

from Demo.satellite_sup_2d.ablation_satellite import get_setting_satellite, get_loaders_satellite_GUT
from Demo.satellite_sup_2d.trains_satellite import train_base_GUT as train_base, valid_base_GUT as valid_base
from Demo.satellite_sup_2d.ablation_satellite import (
    get_loaders_satellite_multi_GUT,
    transform_U_channels
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5', type=str,
    default="/data/wqn/datasets/SDNO_test/15c_data_test.h5")
    parser.add_argument('--ntrain', type=int, default=None, help='训练样本数，默认80%')
    parser.add_argument('--nvalid', type=int, default=None, help='验证样本数，默认剩余')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--work_name', type=str, default='Trans_satellite_super_v2')
    parser.add_argument('--win_split', type=int, default=1)
    parser.add_argument('--shuffle', action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    basic_dict, train_dict, pred_model_dict, super_model_dict = get_setting_satellite()
    if args.batch_size: train_dict['batch_size'] = args.batch_size
    if args.epochs: train_dict['epochs'] = args.epochs
    if args.lr: train_dict['learning_rate'] = args.lr
    if args.ntrain: basic_dict['ntrain'] = args.ntrain
    if args.nvalid: basic_dict['nvalid'] = args.nvalid

    # 工作路径
    work_path = os.path.join('work_satellite', args.work_name)
    work = WorkPrj(work_path)
    Logger = TextLogger(os.path.join(work_path, 'train.log'))
    Device = work.device

    locals().update(basic_dict)
    locals().update(train_dict)
    target_U_channels = 1
    pred_model_dict['node_feats'] = target_U_channels + 4

    # 数据加载（固定V2/17通道）
    train_loader, valid_loader, normalizers, meta = get_loaders_satellite_multi_GUT(
        component_nums=[1, 2, 3, 4, 5],
        target_U_channels=target_U_channels,
        empty_channel_value=1.0,
        train_num=ntrain,
        valid_num=nvalid,
        batch_size=16,
        shuffled=True
    )

    # 网络
    Net_model = FourierTransformer(**pred_model_dict).to(Device)

    # 损失与优化器
    Loss_func_train = nn.MSELoss()
    Loss_func_valid = nn.MSELoss()
    Optimizer_C = torch.optim.Adam(Net_model.parameters(), lr=learning_rate, betas=(0.7, 0.9), weight_decay=1e-7)
    Scheduler_C = torch.optim.lr_scheduler.StepLR(Optimizer_C, step_size=scheduler_step, gamma=scheduler_gamma)

    # 可视化
    Visual = MatplotlibVision(work_path, input_name=('x', 'y'), field_name=('T',))

    # 训练循环
    start_time = time.time()
    log_loss = {
        'train_c_loss': [],
        'valid_c_loss': [],
    }

    for epoch in range(epochs):
        Net_model.train()
        log_loss['train_c_loss'].append(
            train_base(train_loader, Net_model, Device, Loss_func_train, Optimizer_C, Scheduler_C)
        )

        Net_model.eval()
        if valid_loader is not None:
            log_loss['valid_c_loss'].append(
                valid_base(valid_loader, Net_model, Device, Loss_func_valid)
            )

        if epoch % 10 == 0:
            fig, axs = plt.subplots(1, 1, figsize=(15, 8), num=1)
            Visual.plot_loss(fig, axs, np.arange(len(log_loss['train_c_loss'])), np.array(log_loss['train_c_loss']), 'train_c')
            if len(log_loss['valid_c_loss']) > 0:
                Visual.plot_loss(fig, axs, np.arange(len(log_loss['valid_c_loss'])), np.array(log_loss['valid_c_loss']), 'valid_c')
            fig.suptitle('training loss')
            fig.savefig(work.svg)
            plt.close(fig)
            torch.save({'log_loss': log_loss, 'net_model': Net_model.state_dict(), 'optimizer_c': Optimizer_C.state_dict()}, work.pth)
            torch.save(Net_model, work.fpth)

        print('epoch: {:6d}, lr: {:.3e}, train_c: {:.3e}, valid_c: {:.3e}, cost: {:.2f}'.format(
            epoch,
            Optimizer_C.state_dict()['param_groups'][0]['lr'],
            log_loss['train_c_loss'][-1],
            log_loss['valid_c_loss'][-1],
            time.time() - start_time,
        ))
        start_time = time.time()


