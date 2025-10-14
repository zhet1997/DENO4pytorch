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

from transformer.Transformers import FourierTransformer
from Demo.satellite_2d_base.utils import load_yaml_config
from Utilizes.process_data import DataNormer


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
    parser = argparse.ArgumentParser(description='Transformer 卫星热数据训练脚本（与 UNet/FNO/MLP 范式一致）')
    parser.add_argument('--data_path', type=str, default='/data/wqn/turbine_uq/data_post/heat_dataset_780.h5')
    parser.add_argument('--ntrain', type=int, default=None, help='训练样本数；默认按9:1自动分割')
    parser.add_argument('--nvalid', type=int, default=None, help='验证样本数；默认按9:1自动分割')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--cuda_index', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default=os.path.join('work_satellite'))
    # Transformer 关键超参
    parser.add_argument('--hidden', type=int, default=96)
    parser.add_argument('--nhead', type=int, default=2)
    parser.add_argument('--enc_layers', type=int, default=2)
    parser.add_argument('--ffn', type=int, default=None, help='若为None则取 2*hidden')
    parser.add_argument('--decoder', type=str, default='pointwise', choices=['pointwise', 'ifft', 'attention'])
    parser.add_argument('--fourier_modes', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--down', type=int, default=8, help='空间下采样步长，256/down 后作为序列长度根')
    parser.add_argument('--config_path', type=str, default='/data/wqn/DENO4pytorch/data/configs/transformer_config_sate.yml')
    parser.add_argument('--use_config', action='store_true', default=True)
    args = parser.parse_args()

    net_name = 'Trans'
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, f'{net_name}_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)

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

    # 空间下采样，降低注意力 token 数量，避免 OOM
    assert 256 % args.down == 0, 'down 必须整除 256'
    train_np_x = inputs[:ntrain][:, ::args.down, ::args.down, :]
    train_np_y = outputs[:ntrain][:, ::args.down, ::args.down, :]
    valid_np_x = inputs[N - nvalid:][:, ::args.down, ::args.down, :]
    valid_np_y = outputs[N - nvalid:][:, ::args.down, ::args.down, :]

    train_x = torch.tensor(train_np_x, dtype=torch.float32)
    train_y = torch.tensor(train_np_y, dtype=torch.float32)
    valid_x = torch.tensor(valid_np_x, dtype=torch.float32)
    valid_y = torch.tensor(valid_np_y, dtype=torch.float32)

    # 归一化（主程序）
    x_normalizer = DataNormer(train_x.numpy(), method='mean-std')
    y_normalizer = DataNormer(train_y.numpy(), method='mean-std')
    train_x = x_normalizer.norm(train_x)
    valid_x = x_normalizer.norm(valid_x)
    train_y = y_normalizer.norm(train_y)
    valid_y = y_normalizer.norm(valid_y)

    # DataLoader（不丢批）
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(TensorDataset(valid_x, valid_y), batch_size=args.batch_size, shuffle=False, drop_last=False)

    # 组装 Transformer 配置：优先使用 YAML 配置
    cfg = None
    if args.use_config and args.config_path:
        raw_cfg = load_yaml_config(args.config_path) or {}
        # 兼容顶层名称
        if isinstance(raw_cfg, dict) and len(raw_cfg) == 1:
            cfg = list(raw_cfg.values())[0]
        elif isinstance(raw_cfg, dict):
            # 若包含特定键，则取之，否则直接使用
            cfg = raw_cfg.get('PakB_2d', raw_cfg)
        else:
            cfg = {}
        # 强制覆盖与数据维度相关的关键字段
        cfg['node_feats'] = 6
        cfg['n_targets'] = 1
        cfg['pos_dim'] = 2
        cfg['spacial_dim'] = 2
        # feat_extract_type 为空时兜底为 identity
        if cfg.get('feat_extract_type', None) in [None, 'None']:
            cfg['feat_extract_type'] = 'identity'
        # 根据下采样分辨率自动收紧 fourier_modes，避免 RFFT 维度不匹配
        if 'fourier_modes' in cfg:
            s = 256 // args.down
            safe_modes = max(1, min(int(cfg['fourier_modes']), int(s // 2 + 1)))
            cfg['fourier_modes'] = safe_modes
    else:
        cfg = dict(
            node_feats=6,
            n_targets=1,
            n_hidden=args.hidden,
            n_head=args.nhead,
            num_encoder_layers=args.enc_layers,
            dim_feedforward=(2 * args.hidden) if args.ffn is None else args.ffn,
            attention_type='fourier',
            feat_extract_type='identity',
            num_feat_layers=0,
            graph_activation=False,
            raw_laplacian=True,
            pos_dim=2,
            edge_feats=0,
            layer_norm=True,
            attn_norm=False,
            batch_norm=False,
            spacial_residual=False,
            return_attn_weight=False,
            seq_len=None,
            bulk_regression=False,
            decoder_type=args.decoder,
            num_regressor_layers=2,
            fourier_modes=args.fourier_modes,
            freq_dim=64,
            spacial_dim=2,
            spacial_fc=True,
            dropout=args.dropout,
            xavier_init=1e-4,
            diagonal_weight=1e-2,
            symmetric_init=False,
            debug=False,
        )

    Net_model = FourierTransformer(**cfg).to(device)

    # 优化器/调度器/损失
    Loss_func = nn.MSELoss()
    Optimizer = torch.optim.Adam(Net_model.parameters(), lr=args.lr, betas=(0.7, 0.9), weight_decay=1e-4)
    Scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=max(args.epochs // 2, 1), gamma=0.1)
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
    Net_model.eval()
    with torch.no_grad():
        bx, by = next(iter(train_loader))
        bx = bx.to(device)
        gd = feature_transform(bx)
        pred_train0 = Net_model(bx, gd).cpu().numpy()
        by_np = by.numpy()
        pred_train0_den = y_normalizer.back(pred_train0)
        by_den = y_normalizer.back(by_np)
        _plot_compare(by_den[0, ..., 0], pred_train0_den[0, ..., 0], os.path.join(save_dir, 'train_solution_0.jpg'))
        vx, vy = next(iter(valid_loader))
        vx = vx.to(device)
        gdv = feature_transform(vx)
        pred_valid0 = Net_model(vx, gdv).cpu().numpy()
        vy_np = vy.numpy()
        pred_valid0_den = y_normalizer.back(pred_valid0)
        vy_den = y_normalizer.back(vy_np)
        _plot_compare(vy_den[0, ..., 0], pred_valid0_den[0, ..., 0], os.path.join(save_dir, 'valid_solution_0.jpg'))
    torch.save({'net_model': Net_model.state_dict(), 'optimizer': Optimizer.state_dict(), 'epoch': -1},
               os.path.join(save_dir, 'latest_model.pth'))

    # 训练循环
    for epoch in range(args.epochs):
        train_step_loss = train_epoch(train_loader, Net_model, device, Loss_func, Optimizer, Scheduler)
        valid_loss = valid_epoch(valid_loader, Net_model, device, Loss_func)
        log_loss['train'].append(train_step_loss)
        log_loss['valid'].append(valid_loss)
        elapsed = time.time() - start_time
        print(
            f"epoch: {epoch:6d}, lr: {Optimizer.param_groups[0]['lr']:.3e}, "
            f"train_step_loss: {train_step_loss:.3e}, valid_step_loss: {valid_loss:.3e}, cost: {elapsed:.2f}"
        )
        start_time = time.time()

        if (epoch % 5 == 0) or (epoch == args.epochs - 1) or (epoch == 0):
            plt.figure(figsize=(8,4))
            plt.plot(np.arange(len(log_loss['train'])), log_loss['train'], label='train_step')
            plt.plot(np.arange(len(log_loss['valid'])), log_loss['valid'], label='valid_step')
            plt.yscale('log')
            plt.legend(); plt.title('training loss'); plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'loss.svg')); plt.close()

        if (epoch % 100 == 0) or (epoch == args.epochs - 1):
            with torch.no_grad():
                bx, by = next(iter(train_loader))
                bx = bx.to(device)
                gd = feature_transform(bx)
                pred_b = Net_model(bx, gd).cpu().numpy()
                pred_b_den = y_normalizer.back(pred_b)
                _plot_compare(by_den[0, ..., 0], pred_b_den[0, ..., 0], os.path.join(save_dir, f'train_solution_{epoch}.jpg'))
                vx, vy = next(iter(valid_loader))
                vx = vx.to(device)
                gdv = feature_transform(vx)
                pred_v = Net_model(vx, gdv).cpu().numpy()
                pred_v_den = y_normalizer.back(pred_v)
                _plot_compare(vy_den[0, ..., 0], pred_v_den[0, ..., 0], os.path.join(save_dir, f'valid_solution_{epoch}.jpg'))

        if (epoch % 10 == 0) or (epoch == args.epochs - 1):
            torch.save({'net_model': Net_model.state_dict(), 'optimizer': Optimizer.state_dict(),
                        'epoch': epoch, 'log_loss': log_loss}, os.path.join(save_dir, 'latest_model.pth'))


if __name__ == '__main__':
    main()


