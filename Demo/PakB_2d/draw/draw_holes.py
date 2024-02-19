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
# from Tools.model_define.define_STNO import predictor
from Tools.model_define.define_TNO_1 import predictor
def draw_field(true, pred, grid, work=None, save_path=None, parameterList=None):
    ## draw the figure
    Visual = MatplotlibVision(os.path.join(work.root, 'save_figure'), input_name=('x', 'y'), field_name=('none'))
    # Visual.font_EN = {'family': 'Times New Roman', 'weight': 'normal', 'size': 10}
    # Visual.font_CHN = {'family': 'SimSun', 'weight': 'normal', 'size': 10}
    # rangeList = draw_range_dict(parameterList)
    for i in range(32):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        Visual.field_name = ('T',)
        Visual.plot_fields_ms(fig, axs, true[i], pred[i], grid,
                              show_channel=None, cmaps=['Spectral_r', 'Spectral_r', 'coolwarm'],
                              fmin_max=[[310], [323],],
                              limitList=[1.5,],
                              )
        fig.patch.set_alpha(0.)
                              # fmin_max=np.array(rangeList).T)
        fig.savefig(os.path.join(save_path, 'meridian_' + str(i) + '_valid' + '.jpg'), bbox_inches='tight', transparent=True)

def draw_eta_curve(Visual, rst, label=None, xlim=[0,1], fig=None, axs=None):
    # colorList = ['steelblue', 'darkslateblue']
    colorList = ['navy', 'chocolate']
    # colorList = ['g', 'lawngreen']
    markerList = ['^', '-']
    shape = rst.shape
    Visual.plot_curve_scatter(fig, axs,
                              np.tile(np.linspace(0, 1, shape[1]), (shape[0], 1)),
                              rst,
                              labelList=label,
                              colorList=colorList, markerList=markerList, xlim=xlim,
                              xylabels=('', "cooling effectiveness"))
def draw_curves(true, pred, grid, work=None, save_path=None, parameterList=None):
    ## draw the figure
    Visual = MatplotlibVision(os.path.join(work.root, 'save_figure'), input_name=('x', 'y'), field_name=('none'))
    # Visual.font_EN = {'family': 'Times New Roman', 'weight': 'normal', 'size': 10}
    # Visual.font_CHN = {'family': 'SimSun', 'weight': 'normal', 'size': 10}
    # rangeList = draw_range_dict(parameterList)
    rst_true = np.mean(true, axis=1)[:,::2,:]
    rst_true = (323 - rst_true) / (323-286)
    rst_pred = np.mean(pred, axis=1)[:,::2,:]
    rst_pred = (323 - rst_pred) / (323 - 286)
    for i in range(32):
        fig, axs = plt.subplots(1, 1, figsize=(14, 5))
        Visual.field_name = ('T',)

        rst = np.concatenate((rst_true[i:i+1], rst_pred[i:i+1]), axis=0)
        draw_eta_curve(Visual, rst, label=['true', 'pred'], fig=fig, axs=axs)
        fig.savefig(os.path.join(save_path, 'curves' + '_' + str(i) + '.jpg'), bbox_inches='tight')

if __name__ == '__main__':
    os.chdir(r'E:\\WQN\\CODE\\DENO4pytorch/')

    name = 'Trans'
    # name = 'FNO'
    # name = 'TNO_12_self'
    input_dim = 16
    output_dim = 8
    type = 'valid'
    print(os.getcwd())
    work_path = os.path.join('Demo', 'PakB_2d', 'work', name + '_' + str(9) + '_mask_3')
    # work_path = os.path.join('Demo', 'PakB_2d', 'work', name + '_x' + str(0))
    # work_path = os.path.join('Demo', 'PakB_2d', 'work', name + '_' + str(5))
    # work_load_path = os.path.join('Demo', 'PakB_2d', 'work')
    work = WorkPrj(work_path)
    save_path = os.path.join(work.root, 'save_figure')

    if not os.path.exists(work.valid):
        work.save_pred()
    if type=='train':
        data = np.load(work.train)
    elif type=='valid':
        data = np.load(work.valid)
    elif type == 'valid_sim':
        data = np.load(work.valid_sim)

    data_true = data['true']
    data_pred = data['pred']
    data_grid = data['grid']

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    draw_curves(data_true, data_pred, data_grid, work=work, save_path=save_path)
    draw_field(data_true, data_pred, data_grid, work=work, save_path=save_path)