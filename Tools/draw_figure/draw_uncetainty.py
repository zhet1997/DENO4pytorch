import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
torch.cuda.is_available()
from Tools.post_process.model_predict import predictor_establish
from Tools.uncertainty.SensitivityUncertainty import Turbo_UQLab
from Tools.post_process.post_CFD import cfdPost_2d
from Tools.draw_figure.draw_compare import *
from Tools.uncertainty.GVRB_setting import get_evaluate_set, get_problem_set, get_match_dict
from Tools.uncertainty.GVRB_setting import InputTransformer, UQTransformer
from Tools.draw_figure.draw_opt_rst import optiaml_predicter
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def get_uq_0d(data, evaluater=None,
               **kwargs
               ):
    problem = get_problem_set('S1_hub') #会被替换
    evaluate = get_evaluate_set()
    evaluate.update({'evaluate_func': evaluater})
    SA = Turbo_UQLab(problem, evaluate)
    rst = SA.value_pure_evaluate(data)

    return rst

def draw_uq_0d(data, evaluater=None,
               save_path=None,
               parameterList=None,
               uq_name=None,
               colorList=None,
               **kwargs
               ):
    var_name = 'S1'
    rst = get_uq_0d(data, evaluater=evaluater)
    assert len(data.shape)==2
    sam_num = data.shape[0]
    rst = rst.reshape([-1, sam_num,len(parameterList)]).transpose([1,0,2])

    ## draw the figure
    if save_path is None:
        save_path = os.path.join(work.root, 'save_figure')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    Visual = MatplotlibVision('', input_name=('x', 'y'), field_name=('none'))
    labelList = ['baseline', 'optimal_dtm', 'optimal_uq']
    if rst.shape[0] > 3:
        for _ in range(rst.shape[0]-3):
            labelList.append(None)
            colorList.append(colorList[2])
    for j in range(len(parameterList)):
        fig, axs = plt.subplots(1, 1, figsize=(10, 10))
        # Visual.plot_histogram(fig, axs, rst[..., j], bins=20, rangeList=None, color=colorList, alpha=0.6, label=labelList)
        Visual.plot_kde(fig, axs, rst[..., j], bins=20, rangeList=None, color=colorList, alpha=0.6,
                              label=labelList)
        fig.savefig(os.path.join(save_path, 'hist_' + uq_name + '_' + var_name + '_' + parameterList[j] + '.jpg'))
        plt.close(fig)


def draw_uq_0d_multi_bc(data, evaluater=None,
                        save_path=None,
                        var_name=None,
                        uqList =None,
                        parameterList=None,
                        model_all=None,
                        colorList=None,
                        **kwargs
                        ):
    rst_dict = {}
    for uq in uqList:
        adapter_gvrb = UQTransformer(var_name, uq_name=uq, uq_number=1280)
        P = optiaml_predicter(model=model_all,
                              adapter=adapter_gvrb,
                              parameterList=parameterList,
                              )
        evaluater = lambda X: P.evaluate_with_bc_change(X, type='norm')
        rst_dict.update({uq:get_uq_0d(data, evaluater=evaluater)})

    ## draw the figure
    if save_path is None:
        save_path = os.path.join('..', 'save_figure')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    Visual = MatplotlibVision('', input_name=('x', 'y'), field_name=('none'))


    for j in range(len(parameterList)):
        range_min = np.nan
        range_max = np.nan
        fig, axs = plt.subplots(1, 1, figsize=(10, 8))
        data_draw = []
        for k, uq in enumerate(uqList):
            range_min = np.nanmin(np.append(rst_dict[uq][..., j], range_min))
            range_max = np.nanmax(np.append(rst_dict[uq][..., j], range_max))
            data_draw.append(rst_dict[uq][..., j])
        # Visual.plot_histogram(fig, axs, data_draw, bins=30, rangeList=[range_min, range_max], color=colorList[:4],
        #                       alpha=0.5, label=uqList)
        Visual.plot_kde(fig, axs, data_draw, bins=30, rangeList=[range_min, range_max], color=colorList[:4],
                              alpha=0.3, label=uqList, xlabel=parameterList[j])
        # for k, uq in enumerate(uqList):
        #     Visual.plot_histogram(fig, axs, rst_dict[uq][..., j], bins=30, range=[range_min, range_max], color=colorList[k], alpha=0.5, label=uq)
        if len(var_name) == 2:
            var_name = 'S1R1'

        fig.savefig(os.path.join(save_path, 'hist_' + 'all_' + var_name + '_' + parameterList[j] + '.jpg'))
        plt.close(fig)

def get_uq_1d(data, evaluater=None,
              expand=100,
               **kwargs
               ):
    var_name = 'S1_hub'
    problem = get_problem_set(var_name)
    evaluate = get_evaluate_set()
    evaluate.update({'evaluate_func': evaluater})
    SA = Turbo_UQLab(problem, evaluate)
    rst = SA.value_pure_evaluate(data)*100

    return rst
def draw_uq_1d(data,
               evaluater=None,
               work=None,
               parameterList=None,
               save_path=None,
               grid=None,
               **kwargs,
               ):

    rst = get_uq_0d(data, evaluater=evaluater)
    ## draw the figure
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    Visual = MatplotlibVision('', input_name=('x', 'y'), field_name=('none'))

    var_name = 'S1'
    for j in range(len(parameterList)):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        Visual.plot_3d_slices_histogram(rst[:, 12:-12:4,j], hist_bins=200, slice_orientation='z', ax=ax)
        fig.savefig(os.path.join(save_path, 'spanStd_' + var_name + '_' + parameterList[j].replace('/','') + '.jpg'))
        plt.close(fig)

def draw_uq_2d(predictor, work=None):
    evaluater = lambda x: predictor.predictor_cfd_value(x, input_norm=False, parameterList=parameterList, grid=grid,
                                                        space=2)
    match_dict = get_match_dict()
    # # temp = [x for x in match_dict.keys() if not '_' in x]
    for var_name in match_dict.keys():
        ## get the draw data
        problem = get_problem_set(var_name)
        evaluate = get_evaluate_set()
        evaluate.update({'evaluate_func': evaluater})
        SA = Turbo_UQLab(problem, evaluate)
        data = SA.sample_generate(1280, dist='uniform', generate='lhs', paradict=None)
        rst = SA.value_evaluate(data)
        mean = SA.moment_calculate(rst, type=1)
        var = np.sqrt(SA.moment_calculate(rst, type=2))

        ## draw the figure
        save_path = os.path.join(work.root, 'save_figure')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        Visual = MatplotlibVision(os.path.join(work.root, 'save_figure'), input_name=('x', 'y'), field_name=('none'))
        # rangeList = draw_range_dict(parameterList)
        field_name = draw_name_dict(parameterList)
        Visual.field_name = field_name

        fig, axs = plt.subplots(len(parameterList), 2, figsize=(13, 30))
        Visual.plot_fields_ms_2col(fig, axs, mean, var, grid)
        fig.savefig(os.path.join(save_path, 'meridianStd_' + var_name + '.jpg'))
        plt.close(fig)


def plot_3d_slices_histogram(data, slices=10, hist_bins=30, slice_orientation='z'):
    """
    绘制三维空间中的多个切片上的半透明直方图

    :param data: 一个三元组的数组，表示点的(x, y, z)坐标
    :param slices: 切片的数量
    :param hist_bins: 直方图的bins数量
    :param slice_orientation: 切片的方向 ('x', 'y', 或 'z')
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 根据切片方向获取相应的值
    if slice_orientation == 'x':
        slice_values = data[:, 0]
    elif slice_orientation == 'y':
        slice_values = data[:, 1]
    elif slice_orientation == 'z':
        slice_values = data[:, 2]
    else:
        raise ValueError("slice_orientation must be 'x', 'y', or 'z'")

    # 计算切片的边界
    min_val = np.min(slice_values)
    max_val = np.max(slice_values)
    slice_edges = np.linspace(min_val, max_val, slices + 1)

    # 对每个切片绘制直方图
    for i in range(slices):
        # 找到当前切片中的点
        slice_mask = (slice_values >= slice_edges[i]) & (slice_values < slice_edges[i + 1])
        slice_data = data[slice_mask]

        # 绘制直方图
        hist, bins = np.histogram(slice_data[:, (slice_orientation != 'x') * 0 + (slice_orientation != 'y') * 1],
                                  bins=hist_bins)
        ax.bar(bins[:-1], hist, zs=slice_edges[i], zdir=slice_orientation, width=(max_val - min_val) / slices,
               alpha=0.5)

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()



# draw figures include true and pred
if __name__ == '__main__':
    # 测试函数
    data = np.random.normal(0, 1, (1000, 3))
    plot_3d_slices_histogram(data, slices=5, hist_bins=20, slice_orientation='z')








