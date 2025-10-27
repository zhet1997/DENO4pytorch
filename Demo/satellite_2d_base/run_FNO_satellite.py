import os
import sys
import argparse
import time
import logging
import yaml
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 路径注入，支持绝对路径运行
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir, os.pardir))
MODELS_DIR = os.path.join(PROJECT_ROOT, "Models")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if MODELS_DIR not in sys.path:
    sys.path.insert(0, MODELS_DIR)

from fno.FNOs import FNO2d
from Utilizes.process_data import DataNormer
from Demo.satellite_2d_base.dataset_satellite import load_satellite_data
from Utilizes.visual_data import MatplotlibVision



def feature_transform(x):
    shape = x.shape
    batchsize, size_x, size_y = shape[0], shape[1], shape[2]
    gridx = torch.linspace(0, 1, size_x, dtype=torch.float32)
    gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
    gridy = torch.linspace(0, 1, size_y, dtype=torch.float32)
    gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
    return torch.cat((gridx, gridy), dim=-1).to(x.device)


def train(dataloader, netmodel, device, lossfunc, optimizer, scheduler):
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

def inference(dataloader, netmodel, device): # 这个是？？
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
    parser = argparse.ArgumentParser(description='FNO 卫星热数据训练脚本（参照 Rotor37_2d/run_FNO.py 框架）')
    parser.add_argument('--data_path', type=str, default='/data/wqn/datasets/packaged_dataset20251017_6c/heat_dataset.h5')
    parser.add_argument('--ntrain', type=int, default=8000)
    parser.add_argument('--nvalid', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--cuda_index', type=int, default=7)
    parser.add_argument('--modes_x', type=int, default=10)
    parser.add_argument('--modes_y', type=int, default=10)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--padding', type=int, default=8)
    parser.add_argument('--work_dir', type=str, default=os.path.join('work_satellite'))
    args = parser.parse_args()

    net_name = 'FNO'
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    work_path = os.path.join(args.work_dir, f'{net_name}_n{args.ntrain}_{timestamp}')
    os.makedirs(work_path, exist_ok=True)
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(os.path.join(work_path, 'training.log'), mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f'工作路径: {work_path}')
    logger.info(f'训练样本数: {args.ntrain}, 验证样本数: {args.nvalid}')

    # 设备
    if torch.cuda.is_available():
        Device = torch.device(f'cuda:{args.cuda_index}')
    else:
        Device = torch.device('cpu')

    # 加载数据（inputs: N,256,256,6; outputs: N,256,256,1）
    inputs, outputs = load_satellite_data(args.data_path)
    N = inputs.shape[0]

    ntrain = args.ntrain
    nvalid = args.nvalid
    assert ntrain + nvalid <= N, f'ntrain({ntrain}) + nvalid({nvalid}) 超过数据规模 {N}'
    
    down = 4

    train_x = torch.tensor(inputs[:ntrain, ::down, ::down, :], dtype=torch.float32)
    train_y = torch.tensor(outputs[:ntrain, ::down, ::down, :], dtype=torch.float32)
    valid_x = torch.tensor(inputs[N - nvalid:, ::down, ::down, :], dtype=torch.float32)
    valid_y = torch.tensor(outputs[N - nvalid:, ::down, ::down, :], dtype=torch.float32)

    # 归一化（主程序）
    x_normalizer = DataNormer(train_x.numpy(), method='mean-std')
    y_normalizer = DataNormer(train_y.numpy(), method='mean-std')
    train_x = x_normalizer.norm(train_x)
    valid_x = x_normalizer.norm(valid_x)
    train_y = y_normalizer.norm(train_y)
    valid_y = y_normalizer.norm(valid_y)
    
    # 保存归一化器信息
    normalizer_info = {
        'x_mean': x_normalizer.mean.tolist(),
        'x_std': x_normalizer.std.tolist(),
        'y_mean': y_normalizer.mean.tolist(),
        'y_std': y_normalizer.std.tolist(),
        'method': 'mean-std'
    }
    with open(os.path.join(work_path, 'normalizers.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(normalizer_info, f, allow_unicode=True)
    logger.info(f'归一化器已保存: x_mean.shape={x_normalizer.mean.shape}, y_mean.shape={y_normalizer.mean.shape}')

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=False, drop_last=False)
    valid_loader = DataLoader(TensorDataset(valid_x, valid_y), batch_size=args.batch_size, shuffle=False, drop_last=False)

    modes = (args.modes_x, args.modes_y)
    Net_model = FNO2d(in_dim=6, out_dim=1, modes=modes, width=args.width, depth=args.depth, steps=args.steps,
                      padding=args.padding, activation='gelu').to(Device)

    # 训练要素
    Loss_func = nn.MSELoss()
    Optimizer = torch.optim.Adam(Net_model.parameters(), lr=args.lr, betas=(0.7, 0.9), weight_decay=1e-4)
    Scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=int(args.epochs*0.8), gamma=0.1)
    # 初始化日志与可视化（epoch 0 前）
    log_loss = {'train': [], 'valid': []}
    start_time = time.time()
    
    Visual = MatplotlibVision(work_path, input_name=('1', '2', '3', '4', '5', '6'), field_name=('T',))


    # 训练循环（与Rotor37结构一致的输出格式）
    for epoch in range(args.epochs):
        Net_model.train()
        train_loss = train(train_loader, Net_model, Device, Loss_func, Optimizer, Scheduler)
        Net_model.eval()
        valid_loss = valid(valid_loader, Net_model, Device, Loss_func)
        log_loss['train'].append(train_loss)
        log_loss['valid'].append(valid_loss)
        elapsed = time.time() - start_time
        logger.info('epoch: {:6d}, lr: {:.3e}, train_step_loss: {:.3e}, valid_step_loss: {:.3e}, cost: {:.2f}'.
                    format(epoch, Optimizer.param_groups[0]['lr'], train_loss, valid_loss, elapsed))
        start_time = time.time()
            
        if epoch % 5 == 0:
            fig, axs = plt.subplots(1, 1, figsize=(15, 8), num=1)
            Visual.plot_loss(fig, axs, np.arange(len(log_loss['train'])), log_loss['train'], label='train_step')
            Visual.plot_loss(fig, axs, np.arange(len(log_loss['valid'])), log_loss['valid'], label='valid_step')
            fig.suptitle('training loss')
            fig.savefig(os.path.join(work_path, 'log_loss.svg'))
            plt.close(fig)

        ################################################################
        # Visualization
        ################################################################

        if epoch % 50 == 0:
            train_coord, train_grid, train_true, train_pred = inference(train_loader, Net_model, Device)
            valid_coord, valid_grid, valid_true, valid_pred = inference(valid_loader, Net_model, Device)

            train_true = y_normalizer.back(train_true)
            train_pred = y_normalizer.back(train_pred)
            valid_true = y_normalizer.back(valid_true)
            valid_pred = y_normalizer.back(valid_pred)
            
            
            torch.save({'log_loss': log_loss, 'net_model': Net_model.state_dict(), 'optimizer': Optimizer.state_dict()},
                       os.path.join(work_path, 'latest_model.pth'))
            np.save(os.path.join(work_path, 'loss_history.npy'), log_loss)

            for fig_id in range(5):
                fig, axs = plt.subplots(1, 3, figsize=(18, 20), num=2)
                Visual.plot_fields_ms(fig, axs, train_true[fig_id], train_pred[fig_id], None)
                fig.savefig(os.path.join(work_path, f'train_solution_{str(fig_id)}_{str(epoch)}.jpg'))
                plt.close(fig)

            for fig_id in range(5):
                fig, axs = plt.subplots(1, 3, figsize=(18, 20), num=3)
                Visual.plot_fields_ms(fig, axs, valid_true[fig_id], valid_pred[fig_id], None)
                fig.savefig(os.path.join(work_path, f'valid_solution_{str(fig_id)}_{str(epoch)}.jpg'))
                plt.close(fig)
    
    # 训练完成
    logger.info('训练完成!')
    logger.info(f'最终 train_loss: {log_loss["train"][-1]:.6e}, valid_loss: {log_loss["valid"][-1]:.6e}')