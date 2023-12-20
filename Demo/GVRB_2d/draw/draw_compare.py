import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
torch.cuda.is_available()
import numpy as np
import matplotlib.pyplot as plt
from Utilizes.visual_data import MatplotlibVision
from Tools.train_model.train_task_construct import WorkPrj
from Tools.post_process.post_CFD import cfdPost_2d
from Demo.GVRB_2d.utilizes_GVRB import get_grid_interp,get_origin
from Tools.post_process.model_predict import predictor_establish
from Tools.post_process.load_model import loaddata_Sql, get_true_pred

def stage_define(name=None):
    stage_dict={
        'S1': [0,78],
        'R1': [75,127],
        'stage': [0,127]
    }
    if name is not None:
        idx = stage_dict[name]
        rst = {
            'z1': idx[0],
            'z2': idx[1],
        }
        return rst
    else:
        return {}


def draw_span_curve(Visual, rst, label=None, xlim=None, fig=None, axs=None):
    # colorList = ['steelblue', 'darkslateblue']
    colorList = ['brown', 'chocolate']
    # colorList = ['g', 'lawngreen']
    markerList = ['^', '-']
    shape = rst.shape
    Visual.plot_curve_scatter(fig, axs, rst,
                              np.tile(np.linspace(0, 1, shape[1]), (shape[0], 1)), labelList=label,
                              colorList=colorList, markerList=markerList, xlim=xlim,
                              xylabels=('value', "span"))

def draw_range_dict(key):
    range_dict = {
                'Static_pressure_ratio': [0.45,0.65],#[0.3, 0.5],
                'Total_total_efficiency': [0.6, 1],
                'Total_static_efficiency': [0.6, 1],
                'Isentropic_efficiency': [0.6, 1],
                'Degree_reaction': [0.1, 0.4],
                'Relative Total Pressure': [160000, 360000],
                'Absolute Total Pressure': [160000, 360000],
                'Absolute Mach Number': [0, 1],
                'Static Enthalpy': [570000, 720000],
                'atan(Vx/Vz)': [-85, -60],#[-10, 25],
                'atan(Wx/Wz)': [-60, -20],#[55, 75],
                'Absolute_nozzle_pressure_ratio': [1.5, 2.5],
                'Relative_nozzle_pressure_ratio': [1.1, 1.7],
                'Absolute_Enthalpy': [4000, 20000],
                'Relative_Enthalpy': [-5000, 30000],
                }
    if isinstance(key, list):
        rst = [range_dict[x] for x in key]
    else:
        rst = range_dict[key]
    return rst

def draw_name_dict(key):
    range_dict = {
                'Static_pressure_ratio': [0.3, 0.5],
                'Total_total_efficiency': [0.6, 1],
                'Total_static_efficiency': [0.6, 1],
                'Degree_reaction': [0.1, 0.4],
                'Relative Total Pressure': 'Pt[kPa]',
                'Absolute Total Pressure': 'Pt[kPa]',
                'Absolute Mach Number': '$\mathrm{Ma_{is}}$',
                'Static Enthalpy': '$\mathrm{S[J/K]}$',
                'Static Pressure': 'Ps[kPa]',
                'Static Temperature': 'Ts[K]',
                'Absolute Total Temperature': 'Tt[K]',
                'Vx': 'Vx[m/s]',
                'Vy': 'Vy[m/s]',
                'Vz': 'Vz[m/s]',
                'atan(Vx/Vz)': 'atan[deg]',
                'Density': r'$\mathrm{\rho[kg/m^{3}]}$'
                }
    if isinstance(key, list):
        rst = [range_dict[x] for x in key]
    else:
        rst = range_dict[key]
    return rst


def draw_span_all(post_true, post_pred, work=None, save_path=None):
    ## draw the figure
    Visual = MatplotlibVision(os.path.join(work.root, 'save_figure'), input_name=('x', 'y'), field_name=('none'))
    Visual.font_EN = {'family': 'Times New Roman', 'weight': 'normal', 'size': 10}
    Visual.font_CHN = {'family': 'SimSun', 'weight': 'normal', 'size': 10}
    for j in range(len(parameterList)):
        rst_true = post_true.get_field_performance(parameterList[j], type='spanwised', **stage_define(name=stage_name))
        rst_pred = post_pred.get_field_performance(parameterList[j], type='spanwised', **stage_define(name=stage_name))
        for i in range(114, 115):
            fig, axs = plt.subplots(1, 1, figsize=(7, 9))
            plt.cla()
            rst = np.concatenate((rst_true[i:i+1], rst_pred[i:i+1]), axis=0)
            # draw_span_curve(Visual, rst, label=['true', 'pred'], xlim=draw_range_dict(parameterList[j]), fig=fig, axs=axs)
            draw_span_curve(Visual, rst, label=['true', 'pred'], fig=fig, axs=axs)
            fig.savefig(os.path.join(save_path, parameterList[j].replace('/','') + '_' + str(i) +'.jpg'))
def draw_diagnal(post_true, post_pred, work=None, save_path=None):
    ## draw the figure
    Visual = MatplotlibVision(os.path.join(work.root, 'save_figure'), input_name=('x', 'y'), field_name=('none'))
    # Visual.font_EN = {'family': 'Times New Roman', 'weight': 'normal', 'size': 10}
    # Visual.font_CHN = {'family': 'SimSun', 'weight': 'normal', 'size': 10}
    for j in range(len(parameterList)):
        rst_true = post_true.get_performance(parameterList[j], type='averaged', **stage_define(name=stage_name))
        rst_pred = post_pred.get_performance(parameterList[j], type='averaged', **stage_define(name=stage_name))

        fig, axs = plt.subplots(1, 1, figsize=(10, 10))
        Visual.plot_regression_dot(fig, axs, rst_true.squeeze(), rst_pred.squeeze(),
                                   title=parameterList[j], color='tomato', label='TNO')
        fig.savefig(os.path.join(save_path, 'diagnal_' + parameterList[j] + '_valid' + '.jpg'))
def draw_meridian(post_true, post_pred, work=None, save_path=None):
    ## draw the figure
    Visual = MatplotlibVision(os.path.join(work.root, 'save_figure'), input_name=('x', 'y'), field_name=('none'))
    # Visual.font_EN = {'family': 'Times New Roman', 'weight': 'normal', 'size': 10}
    # Visual.font_CHN = {'family': 'SimSun', 'weight': 'normal', 'size': 10}
    rst_true = post_true.get_fields(parameterList)
    rst_pred = post_pred.get_fields(parameterList)
    # rangeList = draw_range_dict(parameterList)
    field_name = draw_name_dict(parameterList)
    for i in range(50, 55):
        fig, axs = plt.subplots(len(parameterList), 3, figsize=(18, 14))
        Visual.field_name = field_name
        Visual.plot_fields_ms(fig, axs, rst_true[i], rst_pred[i], grid,
                              show_channel=None, cmaps=['Spectral_r', 'Spectral_r', 'coolwarm'],
                              fmin_max=None)
        fig.patch.set_alpha(0.)
                              # fmin_max=np.array(rangeList).T)
        fig.savefig(os.path.join(save_path, 'meridian_' + str(i) + '_valid' + '.jpg'))


def draw_simple(train_true, train_pred,valid_true,valid_pred, work=None, save_path=None):
    design, fields, grids = get_origin(type='struct', realpath='E:\WQN\CODE\DENO4pytorch\Demo\GVRB_2d\data/',
                                       quanlityList=["Static Pressure", "Static Temperature", "Density",
                                                     "Vx", "Vy", "Vz",
                                                     'Relative Total Temperature',
                                                     'Absolute Total Temperature'])
    Visual = MatplotlibVision('', input_name=('x', 'y'),
                              field_name=('ps', 'ts', 'rho', 'vx', 'vy', 'vz', 'tt1', 'tt2'))

    for fig_id in range(5):
        fig, axs = plt.subplots(8, 3, figsize=(18, 25), num=2)
        Visual.plot_fields_ms(fig, axs, train_true[fig_id], train_pred[fig_id], grids)
        fig.savefig(os.path.join(work.root, 'train_solution_' + str(fig_id) + '.jpg'))
        plt.close(fig)
    for fig_id in range(5):
        fig, axs = plt.subplots(8, 3, figsize=(18, 25), num=3)
        Visual.plot_fields_ms(fig, axs, valid_true[fig_id], valid_pred[fig_id], grids)
        fig.savefig(os.path.join(work.root, 'valid_solution_' + str(fig_id) + '.jpg'))
        plt.close(fig)
# draw figures include true and pred
if __name__ == '__main__':

    name = 'TNO_1'
    input_dim = 96
    output_dim = 8
    type = 'valid'
    stage_name = 'stage'
    print(os.getcwd())
    work_load_path = os.path.join("..", 'work1')
    work = WorkPrj(os.path.join(work_load_path, name))
    save_path = os.path.join(work.root, 'save_figure')
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
    #                  # 'Static Pressure',
    #                  'Relative Total Pressure',
    #                  'Absolute Total Pressure',
    #                  # 'Rotary Total Pressure',
    #                  # 'Static Temperature',
    #                  # 'Relative Total Temperature',
    #                  # 'Absolute Total Temperature',
    #                  # 'Rotary Total Temperature',
    #                  # 'Vx', 'Vy', 'Vz','|V|','|V|^2','atan(Vx/Vz)',
    #                  # '|V|^2',
    #                  # 'Wx', 'Wy', 'Wz','|W|','|W|^2','atan(Wx/Wz)',
    #                  # '|U|',
    #                  # 'Speed Of Sound',
    #                  # '|Speed Of Sound|^2',
    #                  # 'Relative Mach Number',
    #                  'Absolute Mach Number',
    #                  'Static Enthalpy',
    #                  # 'Density',
    #                  # 'Entropy',
    #                  # 'Static Energy',
    #                  ]

    ## get the train or valid data
    if not os.path.exists(work.train):
        work.save_pred()
    if type=='train':
        data = np.load(work.train)
    else:
        data = np.load(work.valid)
    data_true = data['true']
    data_pred = data['pred']

    ## get the draw data
    grid = get_grid_interp(grid_num_s=64,grid_num_z=128)
    post_true = cfdPost_2d(data=data_true, grid=grid)
    post_pred = cfdPost_2d(data=data_pred, grid=grid)

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    draw_diagnal(post_true, post_pred, work=work, save_path=save_path)
    draw_span_all(post_true, post_pred, work=work, save_path=save_path)
    draw_meridian(post_true, post_pred, work=work, save_path=save_path)


    # #get train and vaild data
    # parameterList = ['Static_pressure_ratio',
    #                  'Total_total_efficiency',
    #                  'Total_static_efficiency',
    #                  'Degree_reaction',
    #                  ]
    # name = 'TNO_5'
    # input_dim = 96
    # output_dim = 8
    # type = 'valid'
    # stage_name = 'stage'
    # ## load the model
    # print(os.getcwd())
    # work_load_path = os.path.join('E:\WQN\CODE\DENO4pytorch\Demo\GVRB_2d\work1')
    # work_load_path1 = os.path.join('E:\WQN\CODE\DENO4pytorch\Demo\GVRB_2d\work1\TNO_5')
    # work = WorkPrj(os.path.join(work_load_path, name))
    # save_path = os.path.join(work.root, 'save_figure')
    # design, fields, grids = get_origin(type='struct', realpath='E:\WQN\CODE\DENO4pytorch\Demo\GVRB_2d\data/',
    #                                    quanlityList=["Static Pressure", "Static Temperature", "Density",
    #                                                  "Vx", "Vy", "Vz",
    #                                                  'Relative Total Temperature',
    #                                                  'Absolute Total Temperature'])
    # train_loader, valid_loader, x_normalizer, y_normalizer = loaddata_Sql(name, 4000, 900, shuffled=True, )
    # x_normalizer.save(work.x_norm)
    # y_normalizer.save(work.y_norm)
    # Net_model, inference, Device, _, _ = \
    #     predictor_establish(name, work_load_path1, predictor=False)
    # train_source, train_true, train_pred = inference(train_loader, Net_model, Device)
    # valid_source, valid_true, valid_pred = inference(valid_loader, Net_model, Device)
    #
    # post_true = cfdPost_2d(data=valid_true, grid=grids)
    # post_pred = cfdPost_2d(data=valid_pred, grid=grids)
    #
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)
    #
    # draw_diagnal(post_true, post_pred, work=work, save_path=save_path)
    # draw_span_all(post_true, post_pred, work=work, save_path=save_path)
    # # draw_meridian(post_true, post_pred, work=work, save_path=save_path)
    # # draw_simple(train_true,train_pred,valid_true,valid_pred,work=work,save_path=save_path)









