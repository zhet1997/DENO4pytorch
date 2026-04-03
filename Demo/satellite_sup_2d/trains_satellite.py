import torch
import numpy as np
import torch.nn as nn
from typing import Dict, Optional
from Tools.model_define.define_FNO import feature_transform
from Tools.pre_process.data_reform import channel_to_instance, fill_channels
from Tools.pre_process.data_reform import little_windows, big_windows
from Demo.satellite_sup_2d.data_loader_satellite import parse_satellite_g_structure


def build_g_meta_from_tensor(G: torch.Tensor) -> Dict:
    if G.shape[-1] == 5:
        return {
            'G_channels': 5,
            'data_variant': 'mc',
            'G_channel_names': ['cooling_sdf_0', 'cooling_sdf_1', 'cooling_temp', 'coord_x', 'coord_y'],
        }
    return {
        'G_channels': G.shape[-1],
        'data_variant': 'legacy',
        'G_channel_names': None,
    }


def split_satellite_g_inputs(G: torch.Tensor, meta: Optional[Dict] = None) -> Dict:
    return parse_satellite_g_structure(G, meta or build_g_meta_from_tensor(G))


class supredictor_list_windows(nn.Module):

    def __init__(self, pred, supercondition, channel_num=16, win_split=1, g_meta: Optional[Dict] = None):
        super(supredictor_list_windows, self).__init__()
        self.pred_net = pred
        self.super_net = supercondition
        self.channel_num = channel_num
        self.win_split = win_split
        self.g_meta = g_meta

    def forward(self, G, U, c_only=False):
        """
        前向传播
        
        Args:
            G: 条件场 [B, H, W, 4]
            U: 源项场 [B, H, W, K]
            c_only: 是否只使用C网络（不使用super_net）
        
        Returns:
            预测场 [B, H, W, 1]
        """
        parsed_G = split_satellite_g_inputs(G, self.g_meta)
        global_G = parsed_G['global_G']

        if c_only:
            return self._forward_c_only(global_G, U)

        super_num = int(np.log2(U.shape[-1] / self.channel_num))
        design_list = channel_to_instance(U, channel_num=self.channel_num, list=True)
        field_list = []

        for design in design_list:
            field_list.append(self.pred_net(global_G, design))

        for _ in range(super_num):
            super_list = []
            field = torch.cat(field_list, dim=-1)
            field_list = channel_to_instance(field, channel_num=2, list=True)
            coords = feature_transform(field)
            for field in field_list:
                field = little_windows(field, num_rows=self.win_split, num_cols=self.win_split)
                coords_new = little_windows(coords, num_rows=self.win_split, num_cols=self.win_split)
                super_list.append(
                    big_windows(self.super_net(field, coords_new), num_rows=self.win_split, num_cols=self.win_split)
                )
            field_list = super_list

        return field_list[0]
    
    def _forward_c_only(self, G, U):
        """
        只使用C网络（pred_net）的前向传播
        
        策略：对每个U分组分别预测，然后平均
        
        Args:
            G: 条件场 [B, H, W, 4]
            U: 源项场 [B, H, W, K]
        
        Returns:
            预测场均值 [B, H, W, 1]
        """
        parsed_G = split_satellite_g_inputs(G, self.g_meta)
        global_G = parsed_G['global_G']
        design_list = channel_to_instance(U, channel_num=self.channel_num, list=True)
        field_list = []

        for design in design_list:
            field_list.append(self.pred_net(global_G, design))
        
        # 平均所有pred_net的输出
        mean_field = torch.stack(field_list, dim=0).mean(dim=0)
        return mean_field


def train_supercondition(dataloader, netmodel, device, lossfunc, optimizer, scheduler, x_norm=None, super_num=1, channel_num=16):
    train_loss = 0
    for batch, (G, U, T) in enumerate(dataloader):
        G = G.to(device)
        U = U.to(device)
        T = T.to(device)

        pred = netmodel(G, U)
        loss = lossfunc(pred, T)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    scheduler.step()
    return train_loss / (batch + 1)


def valid_supercondition(dataloader, netmodel, device, lossfunc, x_norm=None, super_num=1, channel_num=16):
    valid_loss = 0
    with torch.no_grad():
        for batch, (G, U, T) in enumerate(dataloader):
            G = G.to(device)
            U = U.to(device)
            T = T.to(device)
            pred = netmodel(G, U)
            loss = lossfunc(pred, T)
            valid_loss += loss.item()

    return valid_loss / (batch + 1)


def train_base(dataloader, netmodel, device, lossfunc, optimizer, scheduler):
    train_loss = 0
    for batch, (xx, yy) in enumerate(dataloader):
        xx = xx.to(device)
        yy = yy.to(device)
        gd = feature_transform(xx).to(device)

        pred = netmodel(xx, gd)
        loss = lossfunc(pred, yy)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    scheduler.step()
    return train_loss / (batch + 1)


def valid_base(dataloader, netmodel, device, lossfunc):
    valid_loss = 0
    with torch.no_grad():
        for batch, (xx, yy) in enumerate(dataloader):
            xx = xx.to(device)
            yy = yy.to(device)
            gd = feature_transform(xx).to(device)
            pred = netmodel(xx, gd)
            loss = lossfunc(pred, yy)
            valid_loss += loss.item()

    return valid_loss / (batch + 1)

def train_base_GUT(dataloader, netmodel, device, lossfunc, optimizer, scheduler):
    train_loss = 0
    for batch, (G, U, T) in enumerate(dataloader):
        xx = torch.cat((G, U), dim=-1)
        xx = xx.to(device)
        yy = T.to(device)
        gd = feature_transform(xx).to(device)

        pred = netmodel(xx, gd)
        loss = lossfunc(pred, yy)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    scheduler.step()
    return train_loss / (batch + 1)


def valid_base_GUT(dataloader, netmodel, device, lossfunc):
    valid_loss = 0
    with torch.no_grad():
        for batch, (G, U, T) in enumerate(dataloader):
            xx = torch.cat((G, U), dim=-1)
            xx = xx.to(device)
            yy = T.to(device)
            gd = feature_transform(xx).to(device)
            pred = netmodel(xx, gd)
            loss = lossfunc(pred, yy)
            valid_loss += loss.item()

    return valid_loss / (batch + 1)


def valid_detail(dataloader, netmodel, device, lossfunc, x_norm=None, shuffle=False, super_num=1, channel_num=16, hole_num=1, split_num=0):
    assert super_num == 1
    assert split_num <= hole_num
    valid_loss = 0
    with torch.no_grad():
        for batch, (xx, yy) in enumerate(dataloader):
            xx = xx.to(device)
            yy = yy.to(device)

            xx_0 = xx[..., :split_num].detach().clone()
            xx_1 = xx[..., split_num:].detach().clone()
            xx_0 = fill_channels(xx_0, x_norm=x_norm, channel_num=channel_num, shuffle=shuffle)
            xx_1 = fill_channels(xx_1, x_norm=x_norm, channel_num=channel_num, shuffle=shuffle)

            xx = torch.cat((xx_0, xx_1), dim=-1)
            gd = feature_transform(xx).to(device)
            pred = netmodel(xx, gd)
            loss = lossfunc(pred, yy, xx)
            valid_loss += loss.item()

    return valid_loss / (batch + 1)


def inference_draw(dataloader, netmodel, device, x_norm=None, y_norm=None, shuffle=False, super_num=1, channel_num=16):
    input_all, pred_all, true_all = [], [], []
    for batch, (xx, yy) in enumerate(dataloader):
        xx = xx.to(device)
        yy = yy.to(device)
        xx = fill_channels(xx, x_norm=x_norm, channel_num=channel_num * (2 ** super_num), shuffle=True)
        gd = feature_transform(xx).to(device)
        pred = netmodel(xx, gd)

        pred_all.append(y_norm.back(pred).cpu().numpy())
        true_all.append(y_norm.back(yy).cpu().numpy())
        input_all.append(x_norm.back(xx).cpu().numpy())

    return np.concatenate(input_all, axis=0), np.concatenate(true_all, axis=0), np.concatenate(pred_all, axis=0)


