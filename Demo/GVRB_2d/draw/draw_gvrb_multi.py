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
from Tools.draw_figure.draw_multi_model import draw_span_multi, draw_diagnal_multi
from Tools.model_define.define_TNO_1 import predictor


if __name__ == '__main__':
    os.chdir(r'E:\\WQN\\CODE\\DENO4pytorch/')
    name_list = ['DeepONet', 'FNO', 'TNO',]
    input_dim = 100
    output_dim = 8
    type = 'valid'
    stage_name = 'stage'
    print(os.getcwd())
    work_load_path = os.path.join('Demo', 'GVRB_2d', 'work_draw')
    work_list = []
    for name in name_list:
        work_list.append(WorkPrj(os.path.join(work_load_path, name)))
    save_path = os.path.join(work_list[0].root, 'save_figure_diagnal')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    parameterList = [
                     # 'Static_pressure_ratio',
                     # 'Total_total_efficiency',
                     # 'Total_static_efficiency',
                     # 'Degree_reaction',
                     'Mass_flow',
                     'Polytropic_efficiency',
                     'Isentropic_efficiency',
                     'Absolute_total_pressure_ratio',
                     'Relative_nozzle_pressure_ratio',
                     'Absolute_nozzle_pressure_ratio',
                     'Static_temperature_ratio',
                     'Absolute_total_temperature_ratio',
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
    #                  # 'Relative Total Pressure',
    #                  'Absolute Total Pressure',
    #                  'Static Enthalpy',
    #                  'Absolute Mach Number',
    #                  'Relative Mach Number',
    #                  'atan(Vx/Vz)',
    #                  'atan(Wx/Wz)',
    #                  ]

    ## get the train or valid data


    true_list = []
    pred_list = []
    for work in work_list:
        if not os.path.exists(work.valid):
            work.save_pred()
        if type=='train':
            true_list.append(np.load(work.train)['true'])
            pred_list.append(np.load(work.train)['pred'])
        elif type=='valid':
            true_list.append(np.load(work.valid)['true'])
            pred_list.append(np.load(work.valid)['pred'])

    ## get the draw data
    grid = get_grid_interp(grid_num_s=64,grid_num_z=128)
    data = np.concatenate(true_list[0:1] + pred_list, axis=0)
    post = cfdPost_2d(data=data, grid=grid, boundarycondition=None)




    draw_diagnal_multi(post, save_path=save_path, parameterList=parameterList, sam_num=900)
    # draw_span_multi(post, save_path=save_path, parameterList=parameterList, sam_num=900)
