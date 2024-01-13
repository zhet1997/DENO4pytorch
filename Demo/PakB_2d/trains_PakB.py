import torch
import numpy as np
import torch.nn as nn
from Tools.model_define.define_FNO import feature_transform
from Tools.pre_process.data_reform import channel_to_instance, instance_to_half, fill_channels
class supredictor(nn.Module):

    def __init__(self, pred, supercondition, channel_num=16,):

        super(supredictor, self).__init__()
        self.pred_net = pred
        self.super_net = supercondition
        self.channel_num = channel_num

    def forward(self, design, coords):
        batch_size = design.shape[0]
        super_num = int(np.log2(design.shape[-1]/self.channel_num))
        design = channel_to_instance(design, channel_num=self.channel_num, list=False)
        coords_tmp = coords.tile([int(design.shape[0]/batch_size),1,1,1])
        field = self.pred_net(design, coords_tmp.detach())
        for _ in range(super_num):
            field = instance_to_half(field, batch_size=batch_size, list=False)
            coords_tmp = coords.tile([int(field.shape[0] / batch_size), 1, 1, 1])
            field = self.super_net(field, coords_tmp.detach())
        return field

def train_supercondition(dataloader, netmodel, device, lossfunc, optimizer, scheduler, x_norm=None, super_num=1, channel_num=16):
    """
    Args:
        data_loader: output fields at last time step
        netmodel: Network
        lossfunc: Loss function
        optimizer: optimizer
        scheduler: scheduler
    """
    train_loss = 0
    for batch, (xx, yy) in enumerate(dataloader):
        xx = xx.to(device)
        yy = yy.to(device)
        xx = fill_channels(xx, x_norm=x_norm, channel_num=channel_num * (2 ** super_num), device=device, shuffle=True)
        gd = feature_transform(xx)
        gd = gd.to(device)

        pred = netmodel(xx, gd)
        loss = lossfunc(pred, yy)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    scheduler.step()
    return train_loss / (batch + 1)

def valid_supercondition(dataloader, netmodel, device, lossfunc, x_norm=None, super_num=1, channel_num=16):
    """
    Args:
        data_loader: input coordinates
        model: Network
        lossfunc: Loss function
    """
    valid_loss = 0
    with torch.no_grad():
        for batch, (xx, yy) in enumerate(dataloader):
            xx = xx.to(device)
            yy = yy.to(device)
            xx = fill_channels(xx, x_norm=x_norm, channel_num=channel_num * (2 ** super_num), shuffle=True)
            gd = feature_transform(xx)
            gd = gd.to(device)

            pred = netmodel(xx, gd)
            loss = lossfunc(pred, yy, xx)
            valid_loss += loss.item()

    return valid_loss / (batch + 1)

def valid_detail(dataloader, netmodel, device, lossfunc,
                 x_norm=None,
                 super_num=1, channel_num=16, hole_num=1, split_num=0,
                 ):
    """
    Args:
        data_loader: input coordinates
        model: Network
        lossfunc: Loss function
    """
    assert super_num == 1
    assert split_num <= hole_num
    valid_loss = 0
    with torch.no_grad():
        for batch, (xx, yy) in enumerate(dataloader):
            xx = xx.to(device)
            yy = yy.to(device)

            xx_0 = xx[..., :split_num].detach().clone()
            xx_1 = xx[..., split_num:].detach().clone()
            xx_0 = fill_channels(xx_0, x_norm=x_norm, channel_num=channel_num, shuffle=False)
            xx_1 = fill_channels(xx_1, x_norm=x_norm, channel_num=channel_num, shuffle=False)

            xx = torch.cat((xx_0, xx_1), dim=-1)  # now the channel num is 16*(2**super_num)
            gd = feature_transform(xx)
            gd = gd.to(device)

            pred = netmodel(xx, gd)
            loss = lossfunc(pred, yy, xx)
            valid_loss += loss.item()

    return valid_loss / (batch + 1)