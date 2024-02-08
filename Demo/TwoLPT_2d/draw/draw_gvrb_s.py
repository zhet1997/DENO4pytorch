import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
torch.cuda.is_available()
import numpy as np
import matplotlib.pyplot as plt
from Utilizes.visual_data import MatplotlibVision
from Tools.train_model.train_task_construct import WorkPrj
from Tools.post_process.post_CFD import cfdPost_2d
from Demo.TwoLPT_2d.utilizes_GVRB import get_grid_interp1D
from Tools.draw_figure.draw_compare import draw_diagnal, draw_span_all, draw_meridian

import torch.nn as nn
from torch.utils.data import DataLoader
from Utilizes.process_data import DataNormer
from Models.basic.basic_layers import FcnSingle
from Models.transformer.Transformers import FourierTransformer
from Utilizes.geometrics import gen_uniform_grid
import matplotlib.pyplot as plt
import time
import yaml
from Demo.TwoLPT_2d.utilizes_GVRB import get_origin
from Tools.post_process.model_predict import predictor_establish
from Tools.post_process.load_model import loaddata_Sql, get_true_pred

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

def inference(dataloader, netmodel, device):
    """
    Args:
        dataloader: input coordinates
        netmodel: Network
    Returns:
        out_pred: predicted fields
    """
    grid = gen_uniform_grid(torch.tensor(np.zeros([1, 64, 64, 32]))).to(device)
    with torch.no_grad():
        xx, yy = next(iter(dataloader))
        xx = xx.to(device)
        coords = grid.tile([xx.shape[0], 1, 1, 1])
        pred = netmodel(xx, coords)

    # equation = model.equation(u_var, y_var, out_pred)
    return xx.cpu().numpy(), yy.numpy(), pred.cpu().numpy()

if __name__ == "__main__":
    ################################################################
    # configs
    ################################################################
    os.chdir(r'E:\\WQN\\CODE\\DENO4pytorch/')
    name = 'TNO_1'
    input_dim = 76
    output_dim = 32
    type = 'valid'
    stage_name = 'stage'
    print(os.getcwd())
    work_load_path = os.path.join('Demo', 'TwoLPT_2d', 'work_s')
    work = WorkPrj(os.path.join(work_load_path, name))
    save_path = os.path.join(work.root, 'save_figure_new')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    if torch.cuda.is_available():
        Device = torch.device('cuda')
    else:
        Device = torch.device('cpu')

    design, fields, grids = get_origin(type='struct',
                                       realpath=r'E:\WQN\CODE\DENO4pytorch\Demo\TwoLPT_2d\data',
                                       shuffled=False,
                                       getridbad=True
                                       )  # 获取原始数据取原始数据
    in_dim = 76
    out_dim = 32
    ntrain = 1500
    nvalid = 450

    input = design
    input = torch.as_tensor(input, dtype=torch.float)
    output = fields
    output = torch.as_tensor(output, dtype=torch.float)

    train_x = input[:ntrain]
    train_y = output[:ntrain]
    valid_x = input[-nvalid:]
    valid_y = output[-nvalid:]

    x_normalizer = DataNormer(train_x.numpy(), method='mean-std')
    x_normalizer.save(work.x_norm)
    train_x = x_normalizer.norm(train_x)
    valid_x = x_normalizer.norm(valid_x)

    y_normalizer = DataNormer(train_y.numpy(), method='mean-std')
    y_normalizer.save(work.y_norm)
    train_y = y_normalizer.norm(train_y)
    valid_y = y_normalizer.norm(valid_y)

    train_y = torch.cat((train_y[:, :, :64, :], train_y[:, :, 64:128, :], train_y[:, :, 128:192, :], train_y[:, :, 192:, :]), dim=3)
    valid_y = torch.cat((valid_y[:, :, :64, :], valid_y[:, :, 64:128, :], valid_y[:, :, 128:192, :], valid_y[:, :, 192:, :]), dim=3)

    # train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y),
    #                                            batch_size=1500, shuffle=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(valid_x, valid_y),
                                               batch_size=450, shuffle=False, drop_last=True)

    # ################################################################
    # #  Neural Networks
    # ################################################################
    with open(os.path.join('E:\WQN\CODE\DENO4pytorch\Demo\TwoLPT_2d\data\configs/transformer_config_gvrb.yml')) as f:
        config = yaml.full_load(f)
        config = config['TwoLPT_2d']

    # 建立网络
    Tra_model = FourierTransformer(**config).to(Device)
    MLP_model = FcnSingle(planes=(in_dim, 64, 64, config['n_targets']), last_activation=True).to(Device)
    Share_model = FcnSingle(planes=(config['n_targets'], 64, 64, out_dim), last_activation=False).to(Device)
    Net_model = predictor(trunc=Tra_model, branch=MLP_model, share=Share_model, field_dim=out_dim).to(Device)

    checkpoint = torch.load(work.pth, map_location=work.device)
    Net_model.load_state_dict(checkpoint['net_model'])
    Net_model.eval()

    # valid_source, valid_true, valid_pred = inference(valid_loader, Net_model, Device)
    # valid_true = np.concatenate((valid_true[:, :, :, :8], valid_true[:, :, :, 8:16], valid_true[:, :, :, 16:24],
    #                              valid_true[:, :, :, 24:32]), axis=2)
    # valid_pred = np.concatenate((valid_pred[:, :, :, :8], valid_pred[:, :, :, 8:16], valid_pred[:, :, :, 16:24],
    #                              valid_pred[:, :, :, 24:32]), axis=2)
    # # x, true, pred = inference(valid_loader, Net_model, Device)
    #
    # x = x_normalizer.back(valid_source)
    # true = y_normalizer.back(valid_true)
    # pred = y_normalizer.back(valid_pred)
    #
    # save_dict = {}
    # save_dict.update({'x': x})
    # save_dict.update({'true': true})
    # save_dict.update({'pred': pred})
    # np.savez(work.root+'valid.npz', **save_dict)



    # parameterList = ['Static_pressure_ratio',
    #                  'Total_total_efficiency',
    #                  'Total_static_efficiency',
    #                  'Mass_flow',
    #                  ]

    parameterList = ['Degree_reaction',
                     ]

    # parameterList = ['Static_pressure_ratio',
    #                  'Total_total_efficiency',
    #                  'Total_static_efficiency',
    #                  'atan(Vx/Vz)',
    #                  ]

    # parameterList  = ['Static Pressure',
    #                   'Static Temperature',
    #                   'Density',
    #                   'Vx',
    #                   'Vy',
    #                   'Vz',
    #                   ]
    # parameterList = [
    #                  # 'Relative Total Pressure',
    #                  'Absolute Total Pressure',
    #                  'Static Enthalpy',
    #                  'Absolute Mach Number',
    #                  'atan(Vx/Vz)',
    #                  'atan(Wx/Wz)',
    #                  ]

    ## get the train or valid data

    # if not os.path.exists(work.valid):
    #     work.save_pred()
    if type=='train':
        data = np.load(work.train)
    elif type=='valid':
        data = np.load(work.valid)

    data_x = data['x']
    data_true = data['true']
    data_pred = data['pred']

    ## get the draw data
    grid = get_grid_interp1D(grid_num_s=64,grid_num_z=256)
    # post_true = cfdPost_2d(data=data_true, grid=grid, boundarycondition=None)
    # post_pred = cfdPost_2d(data=data_pred, grid=grid, boundarycondition=None)

    post_true = cfdPost_2d(data=data_true, grid=grid, boundarycondition=data_x[:,-4:])
    post_pred = cfdPost_2d(data=data_pred, grid=grid, boundarycondition=data_x[:,-4:])

    # draw_diagnal(post_true, post_pred, work=work, save_path=save_path, parameterList=parameterList, stage_name='stage2',)
    # draw_span_all(post_true, post_pred, work=work, save_path=save_path, parameterList=parameterList)
    draw_meridian(post_true, post_pred, work=work, save_path=save_path, parameterList=parameterList)