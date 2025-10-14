import os
import sys
import argparse
import time
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

# 数据加载
try:
    from Demo.satellite_2d_base.dataset_satellite import load_satellite_data
except ModuleNotFoundError:
    try:
        from satellite_2d_base.dataset_satellite import load_satellite_data
    except ModuleNotFoundError:
        DEMO_DIR = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
        if DEMO_DIR not in sys.path:
            sys.path.insert(0, DEMO_DIR)
        from satellite_2d_base.dataset_satellite import load_satellite_data


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FNO 卫星热数据训练脚本（参照 Rotor37_2d/run_FNO.py 框架）')
    parser.add_argument('--data_path', type=str, default='/data/wqn/turbine_uq/data_post/heat_dataset_new_cleaned.h5')
    parser.add_argument('--ntrain', type=int, default=None, help='训练样本数；默认按9:1自动分割')
    parser.add_argument('--nvalid', type=int, default=None, help='验证样本数；默认按9:1自动分割')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--cuda_index', type=int, default=6)
    parser.add_argument('--modes_x', type=int, default=4)
    parser.add_argument('--modes_y', type=int, default=4)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--padding', type=int, default=8)
    parser.add_argument('--work_dir', type=str, default=os.path.join('work_satellite'))
    args = parser.parse_args()

    net_name = 'FNO'
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    work_path = os.path.join(args.work_dir, f'{net_name}_{timestamp}')
    os.makedirs(work_path, exist_ok=True)

    # 设备
    if args.device.startswith('cuda') and torch.cuda.is_available():
        Device = torch.device(f'cuda:{args.cuda_index}')
        try:
            torch.cuda.set_device(Device)
        except Exception:
            pass
    else:
        Device = torch.device('cpu')
    print(f'设备: {Device}')

    # 加载数据（inputs: N,256,256,6; outputs: N,256,256,1）
    inputs, outputs = load_satellite_data(args.data_path)
    N = inputs.shape[0]

    # 切分：前取训练，后取验证
    if args.ntrain is None or args.nvalid is None:
        ntrain = int(N * 0.9)
        nvalid = N - ntrain
    else:
        ntrain = args.ntrain
        nvalid = args.nvalid
        assert ntrain + nvalid <= N, f'ntrain({ntrain}) + nvalid({nvalid}) 超过数据规模 {N}'

    train_x = torch.tensor(inputs[:ntrain], dtype=torch.float32)
    train_y = torch.tensor(outputs[:ntrain], dtype=torch.float32)
    valid_x = torch.tensor(inputs[N - nvalid:], dtype=torch.float32)
    valid_y = torch.tensor(outputs[N - nvalid:], dtype=torch.float32)

    # 归一化（主程序）
    x_normalizer = DataNormer(train_x.numpy(), method='mean-std')
    y_normalizer = DataNormer(train_y.numpy(), method='mean-std')
    train_x = x_normalizer.norm(train_x)
    valid_x = x_normalizer.norm(valid_x)
    train_y = y_normalizer.norm(train_y)
    valid_y = y_normalizer.norm(valid_y)

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(TensorDataset(valid_x, valid_y), batch_size=args.batch_size, shuffle=False, drop_last=True)

    # 模型与超参（参考Rotor37 FNO）
    modes = (args.modes_x, args.modes_y)
    Net_model = FNO2d(in_dim=6, out_dim=1, modes=modes, width=args.width, depth=args.depth, steps=args.steps,
                      padding=args.padding, activation='gelu').to(Device)

    # 训练要素
    Loss_func = nn.MSELoss()
    Optimizer = torch.optim.Adam(Net_model.parameters(), lr=args.lr, betas=(0.7, 0.9), weight_decay=1e-4)
    Scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=max(args.epochs // 2, 1), gamma=0.1)
    # 初始化日志与可视化（epoch 0 前）
    log_loss = {'train': [], 'valid': []}
    start_time = time.time()

    # 初始可视化与保存
    Net_model.eval()
    with torch.no_grad():
        bx, by = next(iter(train_loader))
        bx = bx.to(Device)
        gd = feature_transform(bx)
        pred_train0 = Net_model(bx, gd).cpu().numpy()
        by_np = by.numpy()
        pred_train0_den = y_normalizer.back(pred_train0)
        by_den = y_normalizer.back(by_np)

        vx, vy = next(iter(valid_loader))
        vx = vx.to(Device)
        gdv = feature_transform(vx)
        pred_valid0 = Net_model(vx, gdv).cpu().numpy()
        vy_np = vy.numpy()
        pred_valid0_den = y_normalizer.back(pred_valid0)
        vy_den = y_normalizer.back(vy_np)

    def _plot_compare(true2d, pred2d, path):
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1); plt.imshow(true2d, cmap='jet'); plt.title('True'); plt.colorbar(fraction=0.046, pad=0.04)
        plt.subplot(1,2,2); plt.imshow(pred2d, cmap='jet'); plt.title('Pred'); plt.colorbar(fraction=0.046, pad=0.04)
        plt.tight_layout(); plt.savefig(path); plt.close()

    _plot_compare(by_den[0, ..., 0], pred_train0_den[0, ..., 0], os.path.join(work_path, 'train_solution_0.jpg'))
    _plot_compare(vy_den[0, ..., 0], pred_valid0_den[0, ..., 0], os.path.join(work_path, 'valid_solution_0.jpg'))
    torch.save({'net_model': Net_model.state_dict(), 'optimizer': Optimizer.state_dict(), 'epoch': -1},
               os.path.join(work_path, 'latest_model.pth'))

    # 训练循环（与Rotor37结构一致的输出格式）
    for epoch in range(args.epochs):
        Net_model.train()
        train_loss = train(train_loader, Net_model, Device, Loss_func, Optimizer, Scheduler)
        Net_model.eval()
        valid_loss = valid(valid_loader, Net_model, Device, Loss_func)
        log_loss['train'].append(train_loss)
        log_loss['valid'].append(valid_loss)
        elapsed = time.time() - start_time
        print('epoch: {:6d}, lr: {:.3e}, train_step_loss: {:.3e}, valid_step_loss: {:.3e}, cost: {:.2f}'.
              format(epoch, Optimizer.param_groups[0]['lr'], train_loss, valid_loss, elapsed))
        start_time = time.time()

        if (epoch % 5 == 0) or (epoch == args.epochs - 1) or (epoch == 0):
            plt.figure(figsize=(8,4))
            plt.plot(np.arange(len(log_loss['train'])), log_loss['train'], label='train_step')
            plt.plot(np.arange(len(log_loss['valid'])), log_loss['valid'], label='valid_step')
            plt.yscale('log')
            plt.legend()
            plt.title('training loss (log scale)')
            plt.tight_layout()
            plt.savefig(os.path.join(work_path, 'loss.svg'))
            plt.close()

        if (epoch % 100 == 0) or (epoch == args.epochs - 1):
            with torch.no_grad():
                bx, by = next(iter(train_loader))
                bx = bx.to(Device)
                gd = feature_transform(bx)
                pred_b = Net_model(bx, gd).cpu().numpy()
                pred_b_den = y_normalizer.back(pred_b)
                _plot_compare(by_den[0, ..., 0], pred_b_den[0, ..., 0], os.path.join(work_path, f'train_solution_{epoch}.jpg'))
                vx, vy = next(iter(valid_loader))
                vx = vx.to(Device)
                gdv = feature_transform(vx)
                pred_v = Net_model(vx, gdv).cpu().numpy()
                pred_v_den = y_normalizer.back(pred_v)
                _plot_compare(vy_den[0, ..., 0], pred_v_den[0, ..., 0], os.path.join(work_path, f'valid_solution_{epoch}.jpg'))

        if (epoch % 10 == 0) or (epoch == args.epochs - 1):
            torch.save({'net_model': Net_model.state_dict(), 'optimizer': Optimizer.state_dict(),
                        'epoch': epoch, 'log_loss': log_loss}, os.path.join(work_path, 'latest_model.pth'))


