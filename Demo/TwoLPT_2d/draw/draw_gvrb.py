import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
torch.cuda.is_available()
import numpy as np
import matplotlib.pyplot as plt
from Utilizes.visual_data import MatplotlibVision
from Tools.train_model.train_task_construct import WorkPrj
from Tools.post_process.post_CFD import cfdPost_2d
from Demo.TwoLPT_2d.utilizes_GVRB import get_grid_interp
from Tools.draw_figure.draw_compare import draw_diagnal, draw_span_all, draw_meridian
# from Tools.model_define.define_STNO import predictor
from Tools.model_define.define_TNO_1 import predictor



if __name__ == '__main__':
    os.chdir(r'E:\\WQN\\CODE\\DENO4pytorch/')
    name = 'TNO_0'
    input_dim = 76
    output_dim = 8
    type = 'valid'
    stage_name = 'stage'
    print(os.getcwd())
    work_load_path = os.path.join('Demo', 'TwoLPT_2d', 'work')
    work = WorkPrj(os.path.join(work_load_path, name))
    save_path = os.path.join(work.root, 'save_figure_new')
    if not os.path.exists(save_path):
        os.mkdir(save_path)



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

    # parameterList = [
    #                  # 'Relative Total Pressure',
    #                  'Absolute Total Pressure',
    #                  'Static Enthalpy',
    #                  'Absolute Mach Number',
    #                  'Relative Mach Number',
    #                  'atan(Vx/Vz)',
    #                  'atan(Wx/Wz)',
    #                  ]

    ## get the train or valid data

    if not os.path.exists(work.valid):
        work.save_pred()
    if type=='train':
        data = np.load(work.train)
    elif type=='valid':
        data = np.load(work.valid)

    # data_x = data['x']
    data_true = data['true']
    data_pred = data['pred']

    ## get the draw data
    grid = get_grid_interp(grid_num_s=64,grid_num_z=128)
    post_true = cfdPost_2d(data=data_true, grid=grid, boundarycondition=None)
    post_pred = cfdPost_2d(data=data_pred, grid=grid, boundarycondition=None)

    # post_true = cfdPost_2d(data=data_true, grid=grid, boundarycondition=data_x[:,-4:])
    # post_pred = cfdPost_2d(data=data_pred, grid=grid, boundarycondition=data_x[:,-4:])



    draw_diagnal(post_true, post_pred, work=work, save_path=save_path, parameterList=parameterList)
    # draw_span_all(post_true, post_pred, work=work, save_path=save_path, parameterList=parameterList)
    # draw_meridian(post_true, post_pred, work=work, save_path=save_path, parameterList=parameterList)