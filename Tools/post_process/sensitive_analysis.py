import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from post_process.model_predict import predictor_establish
from Tools.data_process import SquareMeshGenerator
from Tools.utilizes_rotor37 import get_grid
from Tools.utilizes_draw import plot_flow_std, plot_span_std
from post_process.post_data import Post_2d
from SALib.sample import latin

def mesh_sliced(input_dim, slice_index, elite=None, type='lhsdesign', sample_num=None):
    slice_dim = len(slice_index)
    if elite is None:
        elite = np.ones([input_dim]) * 0.5
    if sample_num is None:
        sample_num = slice_dim * 101
    if type == "lhsdesign":
        slice_grid = LHSdesign(sample_num, slice_dim)
    elif type == "meshdesign":
        slice_grid = SquareMeshdesign(slice_dim)
    sample_grid = np.tile(elite, [slice_grid.shape[0],1])
    sample_grid[:, slice_index] = slice_grid
    return torch.tensor(sample_grid)

def LHSdesign(sam_num, sam_dim):
    problem = {
        'num_vars': sam_dim,  # 参数数量
        'names': [f'x{i}' for i in range(1, sam_dim + 1)],  # 参数名称
        'bounds': [[0, 1]] * sam_dim,  # 参数范围
    }
    samples = latin.sample(problem, sam_num)
    return samples

def SquareMeshdesign(slice_dim, space=None, mesh_size=None):
    if space is None:
        space = np.tile(np.array([0, 1]), [slice_dim, 1])
    if mesh_size is None:
        mesh_size = np.ones([slice_dim]) * 21

    meshgenerator = SquareMeshGenerator(space, mesh_size)
    slice_grid = meshgenerator.get_grid()
    return  slice_grid

def MkdirCheck(file_path):
    isExist = os.path.exists(file_path)
    if not isExist:
        os.mkdir(file_path)

if __name__ == "__main__":
    name = 'TNO_0'
    input_dim = 28
    output_dim = 5
    ## load the model
    work_load_path = os.path.join("..", 'work')
    grid = get_grid(real_path=os.path.join("..", "data"))
    # Net_model, inference, Device, x_normalizer, y_normalizer = \
    #     predictor_establish(name, work_load_path, predictor=False)
    model_all = predictor_establish(name, work_load_path, predictor=True)
    parameterList = [
        # "Efficiency",
        # "EfficiencyPoly",
        # "PressureRatioV",
        # "TemperatureRatioV",
        "PressureLossR",
        # "EntropyStatic",
        # "MachIsentropic",
        # "Load",
    ]
    var_group = list(range(25))
    var_group = [[x] for x in var_group]

    dict_fig = {}
    dict_axs = {}
    for ii, parameter in enumerate(parameterList):
        fig, axs = plt.subplots(5, 5, figsize=(15, 15), num=ii)
        dict_fig.update({parameter: fig})
        dict_axs.update({parameter: axs})

    for idx, var_list in enumerate(var_group):
        sample_grid = mesh_sliced(input_dim, var_list, sample_num=1001)
        pred = model_all.predicter_2d(sample_grid, input_norm=False)
        grid = get_grid(real_path=os.path.join("..", "data"))
        post_pred = Post_2d(pred.detach().cpu().numpy(), grid)

        fig_id = 0
        # save_path = os.path.join(work_path, "sensitive_test")
        save_path = os.path.join("..", "data", "final_fig")
        dict_axs_sub = {}
        x1 = int(idx / 5)
        x2 = (24-idx) % 5

        for parameter in parameterList:
            dict_axs_sub.update({parameter : dict_axs[parameter][x1][x2]})

        xlimList = [
            [0.5, 1.0],
            [-0.04,0.16],
            [1.75, 2.2],
            [0, 115],
        ]
        tt = 1
        # plot_span_std(post_pred, parameterList,
        #               work_path=os.path.join(save_path, "span_std"),
        #               fig_id=idx, rangeIndex=50, singlefile=True, xlim=xlimList[tt],
        #               singlefigure=True, fig_dict=dict_fig, axs_dict=dict_axs_sub)

        # plot_span_std(post_pred, parameterList,
        #               work_path=os.path.join(save_path, "span_std"),
        #               fig_id=idx, rangeIndex=50, singlefile=True,
        #               singlefigure=True, fig_dict=dict_fig, axs_dict=dict_axs_sub)
        plot_flow_std(post_pred, parameterList,
                      work_path=os.path.join(save_path, "flow_std"),
                      fig_id=idx, rangeIndex=40, singlefile=True, xlim=xlimList[tt],
                      singlefigure=True, fig_dict=dict_fig, axs_dict=dict_axs_sub)
        # for ii, parameter in enumerate(parameterList):
        #     fig = dict_fig[parameter]
        #     plt.figure(fig.number)
        #     plt.show()

        # MkdirCheck(os.path.join(save_path, "span_curve"))
        # plot_span_curve(post_pred, parameterList,
        #                 work_path=os.path.join(save_path, "span_curve"),
        #                 fig_id=idx, singlefile=True)
        #
        # MkdirCheck(os.path.join(save_path, "flow_std"))
        # plot_flow_std(post_pred, parameterList,
        #               work_path=os.path.join(save_path, "flow_std"),
        #               fig_id=idx, rangeIndex=50, singlefile=True)
        #
        # MkdirCheck(os.path.join(save_path, "flow_curve"))
        # plot_flow_curve(post_pred, parameterList,
        #                 work_path=os.path.join(save_path, "flow_curve"),
        #                 fig_id=idx, singlefile=True)
    # for parameter in parameterList:
    #     fig = dict_fig[parameter]
    #     plt.figure(fig.number)
    #     plt.subplots_adjust(hspace=0.1, wspace=0.1)
    #     jpg_path = os.path.join(save_path, parameter + "_flow_" + "all_" + '.jpg')
    #     plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    #     fig.savefig(jpg_path)
    #     plt.close(fig)

