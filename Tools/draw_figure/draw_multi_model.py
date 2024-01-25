import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
torch.cuda.is_available()
import numpy as np
import matplotlib.pyplot as plt
from Utilizes.visual_data import MatplotlibVision
from Tools.draw_figure.draw_compare import stage_define, draw_range_dict
from Tools.train_model.train_task_construct import WorkPrj
from Tools.post_process.post_CFD import cfdPost_2d
from Demo.GVRB_2d.utilizes_GVRB import get_grid_interp,get_origin

def draw_span_curve_multi(Visual, rst, label=None, xlim=None, fig=None, axs=None):
    colors = plt.cm.get_cmap('tab10').colors
    colorlist = ['k']
    for jj in range(len(label)-1):
        colorlist.append(colors[jj])
    markerList = ['-', 'o', 'o', 'o']
    shape = rst.shape
    Visual.plot_curve_scatter(fig, axs, rst,
                              np.tile(np.linspace(0, 1, shape[1]), (shape[0], 1)), labelList=label,
                              colorList=colorlist, markerList=markerList,
                              xylabels=('value', "span"),
                              xlim=xlim,
                              ylim=[0,1],
                              )



def draw_span_multi(post,
                    save_path=None,
                    parameterList=None,
                    stage_name='stage',
                    sam_num=None,
                    ):
    ## draw the figure
    assert post.num%sam_num == 0
    Visual = MatplotlibVision(save_path, input_name=('x', 'y'), field_name=('none'))
    Visual.font_EN = {'family': 'Times New Roman', 'weight': 'normal', 'size': 10}
    Visual.font_CHN = {'family': 'SimSun', 'weight': 'normal', 'size': 10}
    for j in range(len(parameterList)):
        rst = post.get_field_performance(parameterList[j], type='spanwised', **stage_define(name=stage_name))
        rst = rst.reshape(post.num//sam_num, sam_num, post.n_1d)
        rst = np.transpose(rst, (1, 0, 2))
        for i in [100,117,122,128,508]:#range(100, 130):
            fig, axs = plt.subplots(1, 1, figsize=(7, 9))
            plt.cla()
            draw_span_curve_multi(Visual, rst[i],
                                  label=['CFD', 'DeepONet', 'FNO', 'TNO',],
                                  xlim=[rst.min(), rst.max()],
                                  fig=fig, axs=axs)
            fig.savefig(os.path.join(save_path, parameterList[j].replace('/','') + '_' + str(i) +'.jpg'))

def draw_diagnal_multi(post,
                 work=None,
                 save_path=None,
                 parameterList=None,
                 stage_name='stage',
                 sam_num=None,
                 ):
    ## draw the figure
    assert post.num % sam_num == 0
    iter = post.num//sam_num - 1
    Visual = MatplotlibVision('', input_name=('x', 'y'), field_name=('none'))
    # Visual.font_EN = {'family': 'Times New Roman', 'weight': 'normal', 'size': 10}
    # Visual.font_CHN = {'family': 'SimSun', 'weight': 'normal', 'size': 10}
    for j in range(len(parameterList)):
        rst = post.get_performance(parameterList[j], type='averaged', **stage_define(name=stage_name))
        rst = rst.reshape(post.num // sam_num, sam_num)
        rst = np.transpose(rst, (1, 0))
        fig, axs = plt.subplots(1, 1, figsize=(10, 10))
        color_list = plt.cm.get_cmap('tab10').colors[:3]
        label_list = ['DeepONet', 'FNO', 'TNO',]
        for ii in range(iter):
            Visual.plot_regression_dot(fig, axs, rst[:,0], rst[:,ii+1],
                                       title=parameterList[j], color=color_list[ii], label=label_list[ii])
        fig.savefig(os.path.join(save_path, 'diagnal_' + parameterList[j] + '_valid' + '.jpg'))


if __name__ == '__main__':


    name = 'TNO_7'
