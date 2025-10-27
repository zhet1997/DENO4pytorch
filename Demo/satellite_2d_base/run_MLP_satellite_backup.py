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
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Utilizes.process_data import DataNormer
from collections import OrderedDict

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


class MLP(nn.Module):
    def __init__(self, layer_mat=None, is_BatchNorm=False):
        super().__init__()
        if layer_mat is None:
            raise ValueError("layer_mat must be provided")
        self.depth = len(layer_mat)
        activation = nn.GELU

        layer_list = []
        for i in range(self.depth - 2):
            layer_list.append((f'layer_{i}', nn.Linear(layer_mat[i], layer_mat[i + 1])))
            if is_BatchNorm:
                layer_list.append((f'batchnorm_{i}', nn.BatchNorm1d(layer_mat[i + 1])))
            layer_list.append((f'activation_{i}', activation()))
        layer_list.append((f'layer_{self.depth - 2}', nn.Linear(layer_mat[-2], layer_mat[-1])))
        self.layers = nn.Sequential(OrderedDict(layer_list))

        
    def forward(self, x):
        return self.layers(x)


def train(dataloader, netmodel, device, lossfunc, optimizer, scheduler):
    train_loss = 0.0
    for batch, (inp, out) in enumerate(dataloader):
        inp = inp.to(device)
        out = out.to(device)
        pred = netmodel(inp)
        loss = lossfunc(pred, out)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    scheduler.step()
    return train_loss / (batch + 1)


def valid(dataloader, netmodel, device, lossfunc):
    valid_loss = 0.0
    with torch.no_grad():
        for batch, (inp, out) in enumerate(dataloader):
            inp = inp.to(device)
            out = out.to(device)
            pred = netmodel(inp)
            loss = lossfunc(pred, out)
            valid_loss += loss.item()
    return valid_loss / (batch + 1)


def main():
    parser = argparse.ArgumentParser(description='MLP 卫星热数据训练脚本（支持下采样）')
    parser.add_argument('--data_path', type=str, default='/data/wqn/turbine_uq/data_post/heat_dataset_780.h5')
    parser.add_argument('--ntrain', type=int, default=None, help='训练样本数；默认按9:1自动分割')
    parser.add_argument('--nvalid', type=int, default=None, help='验证样本数；默认按9:1自动分割')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--cuda_index', type=int, default=0)
    parser.add_argument('--down', type=int, default=8, help='空间下采样倍数，256/down，建议 4/8')
    parser.add_argument('--hidden', type=int, default=1024, help='MLP隐藏层宽度')
    parser.add_argument('--layers', type=int, default=4, help='总层数（含输入输出）最少3')
    parser.add_argument('--work_dir', type=str, default=os.path.join('work_satellite'))
    args = parser.parse_args()

    net_name = 'MLP'
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    work_path = os.path.join(args.work_dir, f'{net_name}_{timestamp}')
    os.makedirs(work_path, exist_ok=True)

    # 设备
    if args.device.startswith('cuda') and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.cuda_index}')
        try:
            torch.cuda.set_device(device)
        except Exception:
            pass
    else:
        device = torch.device('cpu')
    print(f'设备: {device}')

    # 数据加载
    inputs, outputs = load_satellite_data(args.data_path)
    N = inputs.shape[0]

    # 切分
    if args.ntrain is None or args.nvalid is None:
        ntrain = int(N * 0.9)
        nvalid = N - ntrain
    else:
        ntrain = args.ntrain
        nvalid = args.nvalid
        assert ntrain + nvalid <= N, f'ntrain({ntrain}) + nvalid({nvalid}) 超过数据规模 {N}'

    # 下采样
    assert 256 % args.down == 0, 'down 必须整除 256'
    s = 256 // args.down
    train_x = inputs[:ntrain][:, ::args.down, ::args.down, :]          # (ntrain, s, s, 6)
    train_y = outputs[:ntrain][:, ::args.down, ::args.down, :]         # (ntrain, s, s, 1)
    valid_x = inputs[N - nvalid:][:, ::args.down, ::args.down, :]
    valid_y = outputs[N - nvalid:][:, ::args.down, ::args.down, :]

    # 展平为向量
    in_dim = s * s * 6
    out_dim = s * s * 1
    train_x = torch.tensor(train_x.reshape(ntrain, in_dim), dtype=torch.float32)
    train_y = torch.tensor(train_y.reshape(ntrain, out_dim), dtype=torch.float32)
    valid_x = torch.tensor(valid_x.reshape(nvalid, in_dim), dtype=torch.float32)
    valid_y = torch.tensor(valid_y.reshape(nvalid, out_dim), dtype=torch.float32)

    # 归一化
    x_normalizer = DataNormer(train_x.numpy(), method='mean-std')
    y_normalizer = DataNormer(train_y.numpy(), method='mean-std')
    train_x = x_normalizer.norm(train_x)
    valid_x = x_normalizer.norm(valid_x)
    train_y = y_normalizer.norm(train_y)
    valid_y = y_normalizer.norm(valid_y)

    # DataLoader
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(TensorDataset(valid_x, valid_y), batch_size=args.batch_size, shuffle=False, drop_last=True)

    # 网络层定义
    layers = [in_dim]
    for _ in range(max(args.layers - 2, 1)):
        layers.append(args.hidden)
    layers.append(out_dim)
    net = MLP(layer_mat=layers, is_BatchNorm=False).to(device)

    # 训练组件
    loss_fn = nn.MSELoss()
    optim = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.7, 0.9), weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(optim, step_size=max(args.epochs // 2, 1), gamma=0.1)
    # 初始化日志与可视化（epoch 0 前）
    log_loss = {'train': [], 'valid': []}
    start_time = time.time()

    s = 256 // args.down
    def _plot_compare(true2d, pred2d, path):
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1); plt.imshow(true2d, cmap='jet'); plt.title('True'); plt.colorbar(fraction=0.046, pad=0.04)
        plt.subplot(1,2,2); plt.imshow(pred2d, cmap='jet'); plt.title('Pred'); plt.colorbar(fraction=0.046, pad=0.04)
        plt.tight_layout(); plt.savefig(path); plt.close()

    # 初始可视化与保存
    net.eval()
    with torch.no_grad():
        bx, by = next(iter(train_loader))
        bx = bx.to(device)
        pred0 = net(bx).cpu().numpy()
        by_np = by.numpy()
        pred0_den = y_normalizer.back(pred0)
        by_den = y_normalizer.back(by_np)
        _plot_compare(by_den[0].reshape(s, s), pred0_den[0].reshape(s, s), os.path.join(work_path, 'train_solution_0.jpg'))
        vx, vy = next(iter(valid_loader))
        vx = vx.to(device)
        predv0 = net(vx).cpu().numpy()
        vy_np = vy.numpy()
        predv0_den = y_normalizer.back(predv0)
        vy_den = y_normalizer.back(vy_np)
        _plot_compare(vy_den[0].reshape(s, s), predv0_den[0].reshape(s, s), os.path.join(work_path, 'valid_solution_0.jpg'))
    torch.save({'net_model': net.state_dict(), 'optimizer': optim.state_dict(), 'epoch': -1},
               os.path.join(work_path, 'latest_model.pth'))

    # 训练循环
    for epoch in range(args.epochs):
        net.train()
        train_loss = train(train_loader, net, device, loss_fn, optim, sched)
        net.eval()
        valid_loss = valid(valid_loader, net, device, loss_fn)
        log_loss['train'].append(train_loss)
        log_loss['valid'].append(valid_loss)
        elapsed = time.time() - start_time
        print('epoch: {:6d}, lr: {:.3e}, train_step_loss: {:.3e}, valid_step_loss: {:.3e}, cost: {:.2f}'.
              format(epoch, optim.param_groups[0]['lr'], train_loss, valid_loss, elapsed))
        start_time = time.time()

        if (epoch % 5 == 0) or (epoch == args.epochs - 1) or (epoch == 0):
            plt.figure(figsize=(8,4))
            plt.plot(np.arange(len(log_loss['train'])), log_loss['train'], label='train_step')
            plt.plot(np.arange(len(log_loss['valid'])), log_loss['valid'], label='valid_step')
            plt.yscale('log')
            plt.legend(); plt.title('training loss'); plt.tight_layout()
            plt.savefig(os.path.join(work_path, 'loss.svg')); plt.close()

        if (epoch % 100 == 0) or (epoch == args.epochs - 1):
            with torch.no_grad():
                bx, by = next(iter(train_loader))
                bx = bx.to(device)
                pred_b = net(bx).cpu().numpy()
                pred_b_den = y_normalizer.back(pred_b)
                _plot_compare(by_den[0].reshape(s, s), pred_b_den[0].reshape(s, s), os.path.join(work_path, f'train_solution_{epoch}.jpg'))
                vx, vy = next(iter(valid_loader))
                vx = vx.to(device)
                pred_v = net(vx).cpu().numpy()
                pred_v_den = y_normalizer.back(pred_v)
                _plot_compare(vy_den[0].reshape(s, s), pred_v_den[0].reshape(s, s), os.path.join(work_path, f'valid_solution_{epoch}.jpg'))

        if (epoch % 10 == 0) or (epoch == args.epochs - 1):
            torch.save({'net_model': net.state_dict(), 'optimizer': optim.state_dict(),
                        'epoch': epoch, 'log_loss': log_loss}, os.path.join(work_path, 'latest_model.pth'))


if __name__ == '__main__':
    main()


