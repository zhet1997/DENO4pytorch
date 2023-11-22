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

def draw_span_curve(Visual, rst, label=None, xlim=None, fig=None, axs=None):
    colorList = ['steelblue', 'darkslateblue']
    markerList = ['^', '-']
    shape = rst.shape
    Visual.plot_curve_scatter(fig, axs, rst,
                              np.tile(np.linspace(0, 1, shape[1]), (shape[0], 1)), labelList=label,
                              colorList=colorList, markerList=markerList, xlim=xlim,
                              xylabels=('value', "span"))

def draw_range_dict(key):
    range_dict = {
                'Static_pressure_ratio': [0.3, 0.5],
                'Total_total_efficiency': [0.6, 1],
                'Total_static_efficiency': [0.6, 1],
                'Isentropic_efficiency': [0.6, 1],
                'Degree_reaction': [0.1, 0.4],
                'Relative Total Pressure': [160000, 360000],
                'Absolute Total Pressure': [160000, 360000],
                'Absolute Mach Number': [0, 1],
                'Static Enthalpy': [570000, 720000],
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
                }
    if isinstance(key, list):
        rst = [range_dict[x] for x in key]
    else:
        rst = range_dict[key]
    return rst


def draw_span_all(post_true, post_pred, work=None):
    ## draw the figure
    save_path = os.path.join(work.root, 'save_figure')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    Visual = MatplotlibVision(os.path.join(work.root, 'save_figure'), input_name=('x', 'y'), field_name=('none'))
    Visual.font_EN = {'family': 'Times New Roman', 'weight': 'normal', 'size': 10}
    Visual.font_CHN = {'family': 'SimSun', 'weight': 'normal', 'size': 10}
    for j in range(len(parameterList)):
        rst_true = post_true.get_performance(parameterList[j], type='spanwised')
        rst_pred = post_pred.get_performance(parameterList[j], type='spanwised')
        for i in range(100, 102):
            fig, axs = plt.subplots(1, 1, figsize=(4, 9))
            plt.cla()
            rst = np.concatenate((rst_true[i:i+1], rst_pred[i:i+1]), axis=0)
            draw_span_curve(Visual, rst, label=['true', 'pred'], xlim=draw_range_dict(parameterList[j]), fig=fig, axs=axs)
            fig.savefig(os.path.join(save_path, parameterList[j] + '_' + str(i) +'.jpg'))
def draw_diagnal(post_true, post_pred, work=None):
    ## draw the figure
    save_path = os.path.join(work.root, 'save_figure')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    Visual = MatplotlibVision(os.path.join(work.root, 'save_figure'), input_name=('x', 'y'), field_name=('none'))
    # Visual.font_EN = {'family': 'Times New Roman', 'weight': 'normal', 'size': 10}
    # Visual.font_CHN = {'family': 'SimSun', 'weight': 'normal', 'size': 10}
    for j in range(len(parameterList)):
        rst_true = post_true.get_performance(parameterList[j], type='averaged')
        rst_pred = post_pred.get_performance(parameterList[j], type='averaged')

        fig, axs = plt.subplots(1, 1, figsize=(10, 10))
        Visual.plot_regression_dot(fig, axs, rst_true.squeeze(), rst_pred.squeeze(),
                                   title=parameterList[j], color='tomato', label='TNO')
        fig.savefig(os.path.join(save_path, 'diagnal_' + parameterList[j] + '_valid' + '.jpg'))
def draw_meridian(post_true, post_pred, work=None):
    ## draw the figure
    save_path = os.path.join(work.root, 'save_figure')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    Visual = MatplotlibVision(os.path.join(work.root, 'save_figure'), input_name=('x', 'y'), field_name=('none'))
    # Visual.font_EN = {'family': 'Times New Roman', 'weight': 'normal', 'size': 10}
    # Visual.font_CHN = {'family': 'SimSun', 'weight': 'normal', 'size': 10}
    rst_true = post_true.get_fields(parameterList)
    rst_pred = post_pred.get_fields(parameterList)
    rangeList = draw_range_dict(parameterList)
    field_name = draw_name_dict(parameterList)
    for i in range(50, 55):
        fig, axs = plt.subplots(len(parameterList), 3, figsize=(18, 10))
        Visual.field_name = field_name
        Visual.plot_fields_ms(fig, axs, rst_true[i], rst_pred[i], grid,
                              show_channel=None, cmaps=['Spectral_r', 'Spectral_r', 'coolwarm'],
                              fmin_max=np.array(rangeList).T)
        fig.savefig(os.path.join(save_path, 'meridian_' + str(i) + '_valid' + '.jpg'))

# draw figures include true and pred
if __name__ == '__main__':

    name = 'TNO_9'
    input_dim = 100
    output_dim = 8
    type = 'valid'
    # parameterList = ['Static_pressure_ratio',
    #                  'Total_total_efficiency',
    #                  'Total_static_efficiency',
    #                  'Isentropic_efficiency',
    #                  # 'Degree_reaction',
    #                  ]
    parameterList = [
                     # 'Static Pressure',
                     'Relative Total Pressure',
                     'Absolute Total Pressure',
                     # 'Rotary Total Pressure',
                     # 'Static Temperature',
                     # 'Relative Total Temperature',
                     # 'Absolute Total Temperature',
                     # 'Rotary Total Temperature',
                     # 'Vx', 'Vy', 'Vz','|V|','|V|^2','atan(Vx/Vz)',
                     # '|V|^2',
                     # 'Wx', 'Wy', 'Wz','|W|','|W|^2','atan(Wx/Wz)',
                     # '|U|',
                     # 'Speed Of Sound',
                     # '|Speed Of Sound|^2',
                     # 'Relative Mach Number',
                     'Absolute Mach Number',
                     'Static Enthalpy',
                     # 'Density',
                     # 'Entropy',
                     # 'Static Energy',
                     ]

    ## get the train or valid data
    print(os.getcwd())
    work_load_path = os.path.join("..", 'work')
    work = WorkPrj(os.path.join(work_load_path, name))
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
    post_true = cfdPost_2d(data_true, grid)
    post_pred = cfdPost_2d(data_pred, grid)

    draw_span_all(post_true, post_pred, work=work)
    draw_diagnal(post_true, post_pred, work=work)
    draw_meridian(post_true, post_pred, work=work)









