import torch
import numpy as np
import torch.nn as nn
from Tools.model_define.define_FNO import feature_transform
from Tools.pre_process.data_reform import channel_to_instance, instance_to_half, fill_channels
from Tools.pre_process.data_reform import little_windows, big_windows
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

class supredictor_list(nn.Module):

    def __init__(self, pred, supercondition, channel_num=16):

        super(supredictor_list, self).__init__()
        self.pred_net = pred
        self.super_net = supercondition
        self.channel_num = channel_num

    def forward(self, design, coords):
        super_num = int(np.log2(design.shape[-1]/self.channel_num))
        # assert int(design.shape[-1]/self.channel_num)==2**super_num
        design_list = channel_to_instance(design, channel_num=self.channel_num, list=True)
        field_list = []

        for design in design_list:
            field_list.append(self.pred_net(design, coords))

        for _ in range(super_num):
            super_list = []
            field = torch.cat(field_list, dim=-1)
            field_list = channel_to_instance(field, channel_num=2, list=True)
            for field in field_list:
                super_list.append(self.super_net(field, coords))
            field_list = super_list

        return field_list[0]


class supredictor_list_windows(nn.Module):

    def __init__(self, pred, supercondition, channel_num=16):

        super(supredictor_list_windows, self).__init__()
        self.pred_net = pred
        self.super_net = supercondition
        self.channel_num = channel_num

    def forward(self, design, coords):
        super_num = int(np.log2(design.shape[-1]/self.channel_num))
        # assert int(design.shape[-1]/self.channel_num)==2**super_num
        design_list = channel_to_instance(design, channel_num=self.channel_num, list=True)
        field_list = []

        for design in design_list:
            field_list.append(self.pred_net(design, coords))

        for _ in range(super_num):
            super_list = []
            field = torch.cat(field_list, dim=-1)
            field_list = channel_to_instance(field, channel_num=2, list=True)
            for field in field_list:
                field = little_windows(field)
                coords_new = little_windows(coords)
                super_list.append(big_windows(self.super_net(field, coords_new.detach())))
            field_list = super_list

        return field_list[0]

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
        xx = fill_channels(xx, x_norm=x_norm, channel_num=channel_num * (2 ** super_num), shuffle=True)
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
                 shuffle=False,
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
            xx_0 = fill_channels(xx_0, x_norm=x_norm, channel_num=channel_num, shuffle=shuffle)
            xx_1 = fill_channels(xx_1, x_norm=x_norm, channel_num=channel_num, shuffle=shuffle)

            xx = torch.cat((xx_0, xx_1), dim=-1)  # now the channel num is 16*(2**super_num)
            gd = feature_transform(xx)
            gd = gd.to(device)

            pred = netmodel(xx, gd)
            loss = lossfunc(pred, yy, xx)
            valid_loss += loss.item()

    return valid_loss / (batch + 1)


def inference_detail(dataloader, netmodel, device,
                     x_norm=None,
                     shuffle=False,
                     super_num=1, channel_num=16, hole_num=1, split_num=0,
                     ): # 这个是？？
    """
    Args:
        dataloader: input coordinates
        netmodel: Network
    Returns:
        out_pred: predicted fields
    """
    assert super_num == 1
    assert split_num <= hole_num

    with torch.no_grad():
        xx, yy = next(iter(dataloader))
        xx = xx.to(device)
        xx_0 = xx[..., :split_num].detach().clone()
        xx_1 = xx[..., split_num:].detach().clone()
        xx_0 = fill_channels(xx_0, x_norm=x_norm, channel_num=channel_num, shuffle=shuffle)
        xx_1 = fill_channels(xx_1, x_norm=x_norm, channel_num=channel_num, shuffle=shuffle)

        xx = torch.cat((xx_0, xx_1), dim=-1)  # now the channel num is 16*(2**super_num)
        gd = feature_transform(xx)
        gd = gd.to(device)
        pred = netmodel(xx, gd)

    # equation = model.equation(u_var, y_var, out_pred)
    return xx.cpu().numpy(), yy.numpy(), pred.cpu().numpy()