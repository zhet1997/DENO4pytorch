import os
import sys
import argparse
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# 保证可通过绝对路径运行脚本时，仍能导入项目内模块
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir, os.pardir))
MODELS_DIR = os.path.join(PROJECT_ROOT, "Models")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if MODELS_DIR not in sys.path:
    sys.path.insert(0, MODELS_DIR)

# 兼容多种运行方式的导入（绝对/相对）
try:
    from Demo.satellite_2d_base.dataset_satellite import load_satellite_data
except ModuleNotFoundError:
    # 允许直接以路径运行脚本：尝试不带 Demo 前缀
    try:
        from satellite_2d_base.dataset_satellite import load_satellite_data
    except ModuleNotFoundError:
        # 将 Demo 目录加入路径后再尝试
        DEMO_DIR = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
        if DEMO_DIR not in sys.path:
            sys.path.insert(0, DEMO_DIR)
        from satellite_2d_base.dataset_satellite import load_satellite_data

from Utilizes.process_data import DataNormer
try:
    from cnn.ConvNets import UNet2d
except ModuleNotFoundError:
    # 兜底再插入一次 Models 目录
    if MODELS_DIR not in sys.path:
        sys.path.insert(0, MODELS_DIR)
    from cnn.ConvNets import UNet2d


def feature_transform(x: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    batchsize, size_x, size_y = shape[0], shape[1], shape[2]
    gridx = torch.linspace(0, 1, size_x, dtype=torch.float32, device=x.device)
    gridx = gridx.reshape(1, size_x, 1, 1).repeat(batchsize, 1, size_y, 1)
    gridy = torch.linspace(0, 1, size_y, dtype=torch.float32, device=x.device)
    gridy = gridy.reshape(1, 1, size_y, 1).repeat(batchsize, size_x, 1, 1)
    return torch.cat((gridx, gridy), dim=-1)


def train_epoch(dataloader, netmodel, device, lossfunc, optimizer, scheduler):
    netmodel.train()
    total_loss = 0.0
    total_samples = 0
    for batch, (xx, yy) in enumerate(dataloader):
        xx = xx.to(device)
        yy = yy.to(device)
        gd = feature_transform(xx)

        pred = netmodel(xx, gd)
        loss = lossfunc(pred, yy)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = xx.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

    scheduler.step()
    return total_loss / max(total_samples, 1)


def valid_epoch(dataloader, netmodel, device, lossfunc):
    netmodel.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch, (xx, yy) in enumerate(dataloader):
            xx = xx.to(device)
            yy = yy.to(device)
            gd = feature_transform(xx)

            pred = netmodel(xx, gd)
            loss = lossfunc(pred, yy)
            bs = xx.size(0)
            total_loss += loss.item() * bs
            total_samples += bs
    return total_loss / max(total_samples, 1)


def main():
    parser = argparse.ArgumentParser(description="UNet 卫星热数据训练脚本（主程序归一化、前后切分）")
    parser.add_argument("--data_path", type=str, default="/data/wqn/turbine_uq/data_post/heat_dataset_new_cleaned.h5")
    parser.add_argument("--ntrain", type=int, required=True)
    parser.add_argument("--nvalid", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--cuda_index", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default=os.path.join("work_satellite"))
    args = parser.parse_args()

    # 工作目录：带网络名与时间戳
    net_name = "UNet"
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, f"{net_name}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    # 设备
    if args.device.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.cuda_index}")
        try:
            torch.cuda.set_device(device)
        except Exception:
            pass
    else:
        device = torch.device("cpu")
    print(f"设备: {device}")    
    
    # 加载原始数据（不做归一化/增强）
    inputs, outputs = load_satellite_data(args.data_path)
    N = inputs.shape[0]
    assert args.ntrain + args.nvalid <= N, f"ntrain({args.ntrain}) + nvalid({args.nvalid}) 超过数据规模 {N}"

    # 切分：前取训练，后取验证
    train_x = inputs[:args.ntrain].astype(np.float32)
    train_y = outputs[:args.ntrain].astype(np.float32)
    valid_x = inputs[N - args.nvalid:].astype(np.float32) if args.nvalid > 0 else np.empty((0, 256, 256, 6), dtype=np.float32)
    valid_y = outputs[N - args.nvalid:].astype(np.float32) if args.nvalid > 0 else np.empty((0, 256, 256, 1), dtype=np.float32)

    # 形状断言
    assert train_x.shape == (args.ntrain, 256, 256, 6)
    assert train_y.shape == (args.ntrain, 256, 256, 1)
    if args.nvalid > 0:
        assert valid_x.shape == (args.nvalid, 256, 256, 6)
        assert valid_y.shape == (args.nvalid, 256, 256, 1)

    # 转 tensor
    train_x_t = torch.tensor(train_x, dtype=torch.float32)
    train_y_t = torch.tensor(train_y, dtype=torch.float32)
    valid_x_t = torch.tensor(valid_x, dtype=torch.float32) if args.nvalid > 0 else None
    valid_y_t = torch.tensor(valid_y, dtype=torch.float32) if args.nvalid > 0 else None

    # 主程序归一化（与 Demo/Rotor37_2d/run_UNet.py 风格一致）
    x_normalizer = DataNormer(train_x_t.numpy(), method='mean-std')
    y_normalizer = DataNormer(train_y_t.numpy(), method='mean-std')
    train_x_t = x_normalizer.norm(train_x_t)
    train_y_t = y_normalizer.norm(train_y_t)
    if args.nvalid > 0:
        valid_x_t = x_normalizer.norm(valid_x_t)
        valid_y_t = y_normalizer.norm(valid_y_t)

    # DataLoader
    train_loader = DataLoader(TensorDataset(train_x_t, train_y_t), batch_size=args.batch_size, shuffle=True, drop_last=False)
    valid_loader = (
        DataLoader(TensorDataset(valid_x_t, valid_y_t), batch_size=args.batch_size, shuffle=False, drop_last=False)
        if args.nvalid > 0 else None
    )

    # 模型
    in_dim = 6
    out_dim = 1
    Net_model = UNet2d(
        in_sizes=train_x_t.shape[1:],
        out_sizes=train_y_t.shape[1:],
        width=args.width,
        depth=args.depth,
        steps=args.steps,
        activation='gelu',
        dropout=args.dropout,
    ).to(device)

    # 优化器/调度器/损失
    Loss_func = torch.nn.MSELoss()
    Optimizer = torch.optim.Adam(Net_model.parameters(), lr=args.lr, betas=(0.7, 0.9), weight_decay=1e-4)
    Scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=max(args.epochs // 2, 1), gamma=0.1)

    # 初始化日志与可视化（epoch 0 前）
    log_loss = {'train': [], 'valid': []}
    start_time = time.time()

    # 初始可视化与保存
    Net_model.eval()
    with torch.no_grad():
        bx, by = next(iter(train_loader))
        bx = bx.to(device)
        gd = feature_transform(bx)
        pred_train0 = Net_model(bx, gd).cpu().numpy()
        by_np = by.numpy()
        pred_train0_den = y_normalizer.back(pred_train0)
        by_den = y_normalizer.back(by_np)

        if valid_loader is not None:
            vx, vy = next(iter(valid_loader))
            vx = vx.to(device)
            gdv = feature_transform(vx)
            pred_valid0 = Net_model(vx, gdv).cpu().numpy()
            vy_np = vy.numpy()
            pred_valid0_den = y_normalizer.back(pred_valid0)
            vy_den = y_normalizer.back(vy_np)
        else:
            pred_valid0_den, vy_den = None, None

    def _plot_compare(true2d, pred2d, path):
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1); plt.imshow(true2d, cmap='jet'); plt.title('True'); plt.colorbar(fraction=0.046, pad=0.04)
        plt.subplot(1,2,2); plt.imshow(pred2d, cmap='jet'); plt.title('Pred'); plt.colorbar(fraction=0.046, pad=0.04)
        plt.tight_layout(); plt.savefig(path); plt.close()

    # 保存初始预测图（取前1个样本通道0）
    _plot_compare(by_den[0, ..., 0], pred_train0_den[0, ..., 0], os.path.join(save_dir, 'train_solution_0.jpg'))
    if pred_valid0_den is not None:
        _plot_compare(vy_den[0, ..., 0], pred_valid0_den[0, ..., 0], os.path.join(save_dir, 'valid_solution_0.jpg'))

    # 初次保存模型
    torch.save({
        'net_model': Net_model.state_dict(),
        'optimizer': Optimizer.state_dict(),
        'epoch': -1,
    }, os.path.join(save_dir, 'latest_model.pth'))

    # 训练循环
    for epoch in range(args.epochs):
        train_step_loss = train_epoch(train_loader, Net_model, device, Loss_func, Optimizer, Scheduler)
        valid_loss = valid_epoch(valid_loader, Net_model, device, Loss_func) if valid_loader is not None else 0.0
        log_loss['train'].append(train_step_loss)
        log_loss['valid'].append(valid_loss)
        plt.yscale('log')
        elapsed = time.time() - start_time
        print(f"epoch: {epoch:6d}, lr: {Optimizer.param_groups[0]['lr']:.3e}, "
              f"train_step_loss: {train_step_loss:.3e}, valid_step_loss: {valid_loss:.3e}, cost: {elapsed:.2f}")
        start_time = time.time()

        # 每5个epoch绘制loss曲线
        if (epoch % 5 == 0) or (epoch == args.epochs - 1) or (epoch == 0):
            plt.figure(figsize=(8,4))
            plt.plot(np.arange(len(log_loss['train'])), log_loss['train'], label='train_step')
            if len(log_loss['valid']) > 0:
                plt.plot(np.arange(len(log_loss['valid'])), log_loss['valid'], label='valid_step')
            plt.legend(); plt.title('training loss'); plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'loss.svg'))
            plt.close()

        # 每100个epoch保存预测可视化
        if (epoch % 100 == 0) or (epoch == args.epochs - 1):
            Net_model.eval()
            with torch.no_grad():
                bx, by = next(iter(train_loader))
                bx = bx.to(device)
                gd = feature_transform(bx)
                pred_b = Net_model(bx, gd).cpu().numpy()
                pred_b_den = y_normalizer.back(pred_b)
                _plot_compare(by_den[0, ..., 0], pred_b_den[0, ..., 0], os.path.join(save_dir, f'train_solution_{epoch}.jpg'))
                if valid_loader is not None:
                    vx, vy = next(iter(valid_loader))
                    vx = vx.to(device)
                    gdv = feature_transform(vx)
                    pred_v = Net_model(vx, gdv).cpu().numpy()
                    pred_v_den = y_normalizer.back(pred_v)
                    _plot_compare(vy_den[0, ..., 0], pred_v_den[0, ..., 0], os.path.join(save_dir, f'valid_solution_{epoch}.jpg'))

        # 每10个epoch保存模型
        if (epoch % 10 == 0) or (epoch == args.epochs - 1):
            torch.save({
                'net_model': Net_model.state_dict(),
                'optimizer': Optimizer.state_dict(),
                'epoch': epoch,
                'log_loss': log_loss,
            }, os.path.join(save_dir, 'latest_model.pth'))


if __name__ == "__main__":
    main()


