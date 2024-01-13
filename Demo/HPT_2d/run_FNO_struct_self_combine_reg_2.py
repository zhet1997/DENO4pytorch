import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Utilizes.process_data import DataNormer
from basic.basic_layers import FcnSingle
from transformer.Transformers import FourierTransformer
from Utilizes.geometrics import gen_uniform_grid
from Utilizes.visual_data import MatplotlibVision, TextLogger
from Demo.GVRB_2d.utilizes_GVRB import GVRBWeightLoss
from fno.FNOs import FNO2d
from cnn.ConvNets import UNet2d

import matplotlib.pyplot as plt
import time
import yaml
from Demo.HPT_2d.utilizes_HPT import get_origin, SelfSuperviseLoss_var, SelfSuperviseLoss_reg
from Demo.GVRB_2d.train_model_GVRB.model_whole_life import WorkPrj
from Tools.post_process.post_CFD import cfdPost_2d
import warnings

# 禁止显示 UserWarning
warnings.filterwarnings("ignore", category=UserWarning)
class predictor(nn.Module):

    def __init__(self, branch, trunc, share, field_dim):

        super(predictor, self).__init__()

        self.branch_net = branch
        self.trunc_net = trunc
        self.field_net = share
        # self.field_net = nn.Linear(branch.planes[-1], field_dim)


    def forward(self, design, coords):
        """
        forward compute
        :param design: tensor list[(batch_size, ..., operator_dims[0]), (batch_size, ..., operator_dims[1]), ...]
        :param coords: (batch_size, ..., input_dim)
        """

        T = self.trunc_net(coords)
        B = self.branch_net(design)
        T_size = T.shape[1:-1]
        for i in range(len(T_size)):
            B = B.unsqueeze(1)
        B = torch.tile(B, [1, ] + list(T_size) + [1, ])
        feature = B * T
        F = self.field_net(feature)
        return F


def train(dataloader, netmodel, device, lossfunc, optimizer, scheduler):
    """
    Args:
        data_loader: output fields at last time step
        netmodel: Network
        lossfunc: Loss function
        optimizer: optimizer
        scheduler: scheduler
    """
    grid = gen_uniform_grid(torch.tensor(np.zeros([1, 64, 128, 8]))).to(device)
    train_loss = 0
    for batch, (xx, yy) in enumerate(dataloader):
        xx = xx.to(device)
        yy = yy.to(device)
        coords = grid.tile([xx.shape[0], 1, 1, 1])

        pred = netmodel(xx, coords)
        loss = lossfunc(pred, yy)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    scheduler.step()
    return train_loss / (batch + 1)

def train_combine_reg(dataloader_real, dataloader_virtual, dataloader_sim,
                  netmodel, device,
                  lossfunc_1, lossfunc_2, lossfunc_3,
                  optimizer, scheduler,
                    y_norm=None,
                    mu_2=None,
                      ):
    grid = gen_uniform_grid(torch.tensor(np.zeros([1, 64, 128, 8]))).to(device)
    train_loss_1 = 0
    train_loss_2 = 0
    train_loss_3 = 0
    for batch, ((xx1, yy1), (xx2, yy2), (xx3, yy3)) in enumerate(zip(dataloader_real, dataloader_virtual, dataloader_sim)):
        xx1 = xx1.to(device)
        yy1 = yy1.to(device)
        xx2 = xx2.to(device)
        yy2 = yy2.to(device)
        xx3 = xx3.to(device)
        yy3 = yy3.to(device)

        coords = grid.tile([xx1.shape[0] + xx2.shape[0] + xx3.shape[0], 1, 1, 1])
        pred = netmodel(torch.cat((xx1,xx2,xx3),dim=0), coords)
        pred1 = pred[:xx1.shape[0],...]
        pred2 = pred[xx1.shape[0]:xx1.shape[0] + xx2.shape[0],...]
        pred3 = pred[-xx3.shape[0]:, ...]

        loss1 = lossfunc_1(pred1, yy1)
        loss2 = lossfunc_2(pred2, pred3, yy3, y_norm=y_norm)
        loss3 = lossfunc_3(pred3, y_norm=y_norm)
        # mu_2 = 0.4
        mu_3 = 0.1
        # if loss2.detach()>loss1.detach()*10:
        #     loss = loss1 + 0.1 * loss2/loss2.detach()*loss1.detach() + mu_3 * loss3
        # else:
        loss = loss1 + mu_2 * loss2 + mu_3 * loss3
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_1 += loss1.item()
        train_loss_2 += loss2.item()
        train_loss_3 += loss3.item()

    scheduler.step()
    return train_loss_1 / (batch + 1), train_loss_2 / (batch + 1),  train_loss_3 / (batch + 1)


def valid(dataloader, netmodel, device, lossfunc):
    """
    Args:
        data_loader: input coordinates
        model: Network
        lossfunc: Loss function
    """
    grid = gen_uniform_grid(torch.tensor(np.zeros([1, 64, 128, 8]))).to(device)
    valid_loss = 0
    with torch.no_grad():
        for batch, (xx, yy) in enumerate(dataloader):
            xx = xx.to(device)
            yy = yy.to(device)
            coords = grid.tile([xx.shape[0], 1, 1, 1])
            pred = netmodel(xx, coords)
            loss = lossfunc(pred, yy)
            valid_loss += loss.item()

    return valid_loss / (batch + 1)


def inference(dataloader, netmodel, device):
    """
    Args:
        dataloader: input coordinates
        netmodel: Network
    Returns:
        out_pred: predicted fields
    """
    grid = gen_uniform_grid(torch.tensor(np.zeros([1, 64, 128, 8]))).to(device)
    with torch.no_grad():
        xx, yy = next(iter(dataloader))
        xx = xx.to(device)
        coords = grid.tile([xx.shape[0], 1, 1, 1])
        pred = netmodel(xx, coords)

    # equation = model.equation(u_var, y_var, out_pred)
    return xx.cpu().numpy(), yy.numpy(), pred.cpu().numpy()

def generate_virtual_loader_new(x_normalizer, virtual_batchs, batch_size,
                            scale=[-0.04, 0.04],
                            sim_multi=1,
                            ):
    data_v = x_normalizer.sample_generate(virtual_batchs*batch_size, 2, norm=False)

    post = cfdPost_2d()
    post.bouCondition_data_readin(
        boundarycondition=data_v,
        boundarydict=boundaryDict,
    )

    field_matrix, bc_matrix = post.get_dimensional_matrix(expand=sim_multi, scale=scale)
    field_matrix = np.power(10, field_matrix)
    data_s = post.data_similarity_operate(data_v.copy(), bc_matrix)
    data_v = x_normalizer.norm(data_v)
    data_s = x_normalizer.norm(data_s)

    field_matrix = np.tile(field_matrix[:, None, None, :], [1, 64, 128, 1])
    data_s = np.tile(data_s[:, None, None, :], [1, 64, 128, 1])
    data_v = np.tile(data_v[:, None, None, :], [1, 64, 128, 1])
    field_matrix = torch.as_tensor(field_matrix, dtype=torch.float)
    data_s = torch.as_tensor(data_s, dtype=torch.float)
    data_v = torch.as_tensor(data_v, dtype=torch.float)

    virtual_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data_v, field_matrix),
                                              batch_size=batch_size, shuffle=False, drop_last=True)
    similar_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data_s, field_matrix),
                                              batch_size=batch_size, shuffle=False, drop_last=True)
    return virtual_loader, similar_loader



if __name__ == "__main__":
    ################################################################
    # configs
    ################################################################
    name = 'FNO'
    work_path = os.path.join('work', name + '_' + str(4) + '_self_combine_reg_5.3.x_10')
    train_path = os.path.join(work_path)
    isCreated = os.path.exists(work_path)
    if not isCreated:
        os.makedirs(work_path)
    # 将控制台的结果输出到log文件
    Logger = TextLogger(os.path.join(train_path, 'train.log'))

    if torch.cuda.is_available():
        Device = torch.device('cuda')
    else:
        Device = torch.device('cpu')

    in_dim = 5
    out_dim = 8


    batch_size = 64
    batch_number = 5

    ntrain = batch_number * batch_size
    nvalid = 640

    epochs = 2001
    learning_rate = 0.0002
    scheduler_step = 200
    scheduler_gamma = 0.9

    # net setting
    modes = (10, 10)
    width = 64
    depth = 6
    steps = 1
    padding = 8
    dropout = 0

    boundaryDict = {'Flow Angle': 0,
                    'Absolute Total Temperature': 1,
                    'Absolute Total Pressure': 2,
                    'Rotational_speed': 3,
                    'Shroud_Gap': 4
                    }
    print(epochs, learning_rate, scheduler_step, scheduler_gamma)
    # ################################################################
    # # load data
    # ################################################################


    design, fields, grids = get_origin(type='struct',
                                       realpath='E:\WQN\CODE\DENO4pytorch\Demo\HPT_2d\data')  # 获取原始数据取原始数据
    work = WorkPrj(work_path)
    input = design
    input = np.tile(design[:, None, None, :], (1, 64, 128, 1))
    input = torch.tensor(input, dtype=torch.float)

    # output = fields[:, 0, :, :, :].transpose((0, 2, 3, 1))
    output = fields
    output = torch.tensor(output, dtype=torch.float)

    print(input.shape, output.shape)
    #
    train_x = input[:ntrain]
    train_y = output[:ntrain]
    valid_x = input[-nvalid:]
    valid_y = output[-nvalid:]

    x_normalizer = DataNormer(train_x.numpy(), method='mean-std')
    x_normalizer.dim_change(2)

    y_normalizer = DataNormer(train_y.numpy(), method='mean-std')
    y_normalizer.dim_change(2)

    delta_normalizer = DataNormer(train_y.numpy(), method='mean-std', axis=(1,2))
    delta = train_y.numpy().copy()
    delta = ((delta-np.tile(delta_normalizer.mean[:,None,None,:],[1,64,128,1]))/
             np.tile(delta_normalizer.std[:,None,None,:],[1,64,128,1]))
    delta = np.std(delta, axis=0)


    #########################################################################################################
    # self-supervise data genration
    virtual_batchs = batch_number
    ##########################################################################################################
    # x_normalizer = DataNormer(train_x.numpy(), method='mean-std')
    train_x = x_normalizer.norm(train_x)
    valid_x = x_normalizer.norm(valid_x)

    # y_normalizer = DataNormer(train_y.numpy(), method='mean-std')
    train_y = y_normalizer.norm(train_y)
    valid_y = y_normalizer.norm(valid_y)
    #
    x_normalizer.save(os.path.join(work_path, 'x_norm.pkl'))  # 将normalizer保存下来
    y_normalizer.save(os.path.join(work_path, 'y_norm.pkl'))
    #
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y),
                                               batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(valid_x, valid_y),
                                               batch_size=batch_size, shuffle=False, drop_last=True)


    #
    # ################################################################
    # #  Neural Networks
    # ################################################################
    #
    if name == 'FNO':
        Net_model = FNO2d(in_dim=in_dim, out_dim=out_dim, modes=modes, width=width, depth=depth, steps=steps,
                          padding=padding, activation='gelu').to(Device)
    elif name == 'UNet':
        Net_model = UNet2d(in_sizes=train_x.shape[1:], out_sizes=train_y.shape[1:], width=width,
                           depth=depth, steps=steps, activation='gelu', dropout=dropout).to(Device)





    # model_statistics = summary(Net_model, input_size=(batch_size, train_x.shape[1]), device=str(Device))
    # Logger.write(str(model_statistics))
    #
    # # 损失函数
    Loss_func = nn.MSELoss()
    # Loss_func = GVRBWeightLoss(4, 10, 71)
    Loss_self = SelfSuperviseLoss_var()
    Loss_reg = SelfSuperviseLoss_reg(torch.as_tensor(delta*0.7,dtype=torch.float, device=Device))
    # # Loss_func = nn.SmoothL1Loss()
    # # 优化算法
    Optimizer = torch.optim.AdamW(Net_model.parameters(), lr=learning_rate, betas=(0.7, 0.9), weight_decay=1e-6)
    # # 下降策略
    Scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    # # 可视化
    Visual = MatplotlibVision(work_path, input_name=('x', 'y'), field_name=('ps', 'ts', 'rho', 'vx', 'vy', 'vz', 'tt1', 'tt2'))


    star_time = time.time()

    isExist = os.path.exists(work.pth)
    log_loss = [[], [], [], [], []]
    if isExist:
        print(work.pth)
        checkpoint = torch.load(work.pth, map_location=Device)
        Net_model.load_state_dict(checkpoint['net_model'])
        # Optimizer = checkpoint['log_loss']
        log_loss = checkpoint['log_loss']
        Net_model.eval()

    ################################################################
    # train process
    ################################################################
    # grid = get_grid()
    # grid_real = get_grid()
    # grid = gen_uniform_grid(train_y[:1]).to(Device)
    post = cfdPost_2d()
    valid_loader_sim = post.loader_similarity(valid_loader,
                                                grid=grids, scale=[-0.03, 0.03], expand=1,
                                                boundarydict=boundaryDict,
                                                x_norm=x_normalizer,
                                                y_norm=y_normalizer,
                                                )


    for epoch in range(epochs):

        if epoch==epochs-1:
            torch.save(Net_model,os.path.join(work_path, 'final_model.pth'))

        Net_model.eval()
        log_loss[1].append(valid(valid_loader, Net_model, Device, Loss_func))
        log_loss[4].append(valid(valid_loader_sim, Net_model, Device, Loss_func)) # valid在前更加合理

        Net_model.train()
        # log_loss[0].append(train(train_loader, Net_model, Device, Loss_func, Optimizer, Scheduler))
        if epoch % 10==0:
            self_loader, simlar_loader = generate_virtual_loader_new(x_normalizer, virtual_batchs, batch_size)

        if log_loss[0][-1] > 2e-4:
            temp = 0.5
        else:
            temp = (epoch / epochs) * 5+1
        # log_loss[2].append(train_self_supervise(self_loader, Net_model, Device, Loss_self, Optimizer_self, Scheduler, y_norm=y_normalizer))
        loss1, loss2, loss3 = train_combine_reg(train_loader, self_loader, simlar_loader,
                                         Net_model, Device,
                                         Loss_func, Loss_self, Loss_reg,
                                         Optimizer, Scheduler,
                                         y_norm=y_normalizer,
                                                mu_2=temp
                                                )
        # del self_loader
        log_loss[0].append(loss1)
        log_loss[2].append(loss2)
        log_loss[3].append(loss3)


        print('epoch: {:6d}, lr: {:.3e}, '
              'train_step_loss: {:.3e}, '
              'valid_step_loss: {:.3e}, '
              'self_step_loss: {:.3e}, '
              'reg_step_loss: {:.3e}, '
              'valid_sim_step_loss: {:.3e}, '
              'cost: {:.2f}'.
              format(epoch, learning_rate,
                     log_loss[0][-1],
                     log_loss[1][-1],
                     log_loss[2][-1],
                     log_loss[3][-1],
                     log_loss[4][-1],
                     time.time() - star_time))

        star_time = time.time()

        if epoch > 0 and epoch % 10 == 0:
            torch.save(
                {'log_loss': log_loss, 'net_model': Net_model.state_dict(), 'optimizer': Optimizer.state_dict()},
                os.path.join(work_path, 'latest_model.pth'))

            fig, axs = plt.subplots(1, 1, figsize=(15, 8), num=1)
            Visual.plot_loss(fig, axs, np.arange(len(log_loss[0])), np.array(log_loss)[0, :], 'train_step')
            Visual.plot_loss(fig, axs, np.arange(len(log_loss[0])), np.array(log_loss)[1, :], 'valid_step')
            Visual.plot_loss(fig, axs, np.arange(len(log_loss[0])), np.array(log_loss)[2, :], 'self_step')
            Visual.plot_loss(fig, axs, np.arange(len(log_loss[0])), np.array(log_loss)[4, :], 'valid_2_step')
            fig.suptitle('training loss')
            fig.savefig(os.path.join(train_path, 'log_loss.svg'))
            plt.close(fig)

        ################################################################
        # Visualization
        ################################################################
        if epoch > 0 and epoch % 100 == 0:
            train_source, train_true, train_pred = inference(train_loader, Net_model, Device)
            valid_source, valid_true, valid_pred = inference(valid_loader, Net_model, Device)

            for fig_id in range(5):
                fig, axs = plt.subplots(out_dim, 3, figsize=(18, 20), num=2)
                Visual.plot_fields_ms(fig, axs, train_true[fig_id], train_pred[fig_id], grids)
                fig.savefig(os.path.join(work_path, 'train_solution_' + str(fig_id) + '.jpg'))
                plt.close(fig)

            for fig_id in range(5):
                fig, axs = plt.subplots(out_dim, 3, figsize=(18, 20), num=3)
                Visual.plot_fields_ms(fig, axs, valid_true[fig_id], valid_pred[fig_id], grids)
                fig.savefig(os.path.join(work_path, 'valid_solution_' + str(fig_id) + '.jpg'))
                plt.close(fig)



