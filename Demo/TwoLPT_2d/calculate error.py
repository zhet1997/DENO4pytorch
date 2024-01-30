import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
torch.cuda.is_available()
import numpy as np
import matplotlib.pyplot as plt
from Utilizes.visual_data import MatplotlibVision
from Tools.train_model.train_task_construct import WorkPrj
from Tools.post_process.post_CFD import cfdPost_2d
from Demo.GVRB_2d.utilizes_GVRB import get_grid_interp
from Tools.draw_figure.draw_compare import draw_diagnal, draw_span_all, draw_meridian
# from Tools.model_define.define_STNO import predictor
from Tools.model_define.define_TNO_1 import predictor
from Utilizes.loss_metrics import FieldsLpLoss


if __name__ == '__main__':
    os.chdir(r'E:\\WQN\\CODE\\DENO4pytorch/')


    name = 'deepONet_1'
    input_dim = 100
    output_dim = 8
    type = 'valid'
    stage_name = 'stage'
    print(os.getcwd())
    work_load_path = os.path.join('Demo', 'GVRB_2d', 'work1')
    work = WorkPrj(os.path.join(work_load_path, name))
    # save_path = os.path.join(work.root, 'save_figure_sim')
    # parameterList = [
    #      # 'Absolute_nozzle_pressure_ratio',
    #      # 'Relative_nozzle_pressure_ratio',
    #     # 'Relative Total Pressure',
    #     # 'Absolute Total Pressure',
    #      # 'Absolute_Enthalpy',
    #      # 'Relative_Enthalpy',
    # ]
    parameterList = ['Static_pressure_ratio',
                     'Total_total_efficiency',
                     'Total_static_efficiency',
                     'Degree_reaction',
                     ]
    #                  'atan(Vx/Vz)',
    #                  'atan(Wx/Wz)',
    # 'Mass_flow',
    #                  ]
    # parameterList = [
    #     "Static Pressure",
    #     "Vx", "Vy", "Vz",
    #     'Absolute Total Temperature',
    #     "Density",
    # ]
    # parameterList = [
    #                  'Relative Total Pressure',
    #                  'Absolute Total Pressure',
    #                  'Absolute Mach Number',
    #                  'Static Enthalpy',
    #                  ]

    ## get the train or valid data

    if not os.path.exists(work.valid_sim):
        work.save_pred()
    if type=='train':
        data = np.load(work.train)
    elif type=='valid':
        data = np.load(work.valid)
    elif type == 'valid_sim':
        data = np.load(work.valid_sim)

    data_x = data['x']
    data_true = data['true']
    data_pred = data['pred']

    # ## get the draw data
    # grid = get_grid_interp(grid_num_s=64,grid_num_z=128)
    # post_true = cfdPost_2d(data=data_true, grid=grid, boundarycondition=None)
    # post_pred = cfdPost_2d(data=data_pred, grid=grid, boundarycondition=None)
    #
    # # post_true = cfdPost_2d(data=data_true, grid=grid, boundarycondition=data_x[:,-4:])
    # # post_pred = cfdPost_2d(data=data_pred, grid=grid, boundarycondition=data_x[:,-4:])
    #
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)
    #
    # draw_diagnal(post_true, post_pred, work=work, save_path=save_path, parameterList=parameterList)
    # draw_span_all(post_true, post_pred, work=work, save_path=save_path, parameterList=parameterList)
    # # draw_meridian(post_true, post_pred, work=work, save_path=save_path, parameterList=parameterList)

    Loss_real = FieldsLpLoss(p=2, d=2)

    # # train_source, train_true, train_pred = inference(train_loader, Net_model, Device)
    # # valid_source, valid_true, valid_pred = inference(valid_loader, Net_model, Device)
    # # valid_ex_source, valid_ex_true, valid_ex_pred = inference(valid_expand_loader, Net_model, Device)
    # train_source, train_coord, train_true, train_pred = inference(train_loader, Net_model, Device)
    # valid_source, valid_coord, valid_true, valid_pred = inference(valid_loader, Net_model, Device)
    # train_pred = y_normalizer.back(train_pred)
    # train_true = y_normalizer.back(train_true)
    # valid_pred = y_normalizer.back(valid_pred)
    # valid_true = y_normalizer.back(valid_true)
    # # valid_ex_pred = y_normalizer.back(valid_ex_pred)
    # # valid_ex_true = y_normalizer.back(valid_ex_true)

    absloss = Loss_real.abs(data_true, data_pred)
    relloss = Loss_real.rel(data_true, data_pred)

    # 计算 absloss 和 relloss 的列平均值
    mean_absloss = np.mean(absloss, axis=0)
    mean_relloss = np.mean(relloss, axis=0)

    print("Mean absloss:", mean_absloss)
    print("Mean relloss:", mean_relloss)