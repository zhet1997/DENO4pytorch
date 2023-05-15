import os

import numpy as np
import torch
from post_process.load_model import loaddata, rebuild_model, build_model_yml
from Utilizes.process_data import DataNormer, MatLoader, SquareMeshGenerator
from Demo.Rotor37_2d.utilizes_rotor37 import get_grid, get_origin
from draw_figure import plot_span_std, plot_span_curve, plot_field_2d
from Utilizes.visual_data import MatplotlibVision
import matplotlib.pyplot as plt
from post_process.post_data import Post_2d


if __name__ == "__main__":
    AlgoNameList = ["FNO", "ploynomial", "MLP", "SVR", "XGBoost", "GPR"]
    n_trainList = [500, 1000, 1500, 2000, 2500]
    npz_path = os.path.join("..", "data", "surrogate_data", "scalar_value.npz")
    data_true = np.load(npz_path)

    parameterList = [
        # "PressureRatioV", "TemperatureRatioV",
        # "Efficiency", "EfficiencyPoly",
        # "PressureLossR", "EntropyStatic",
        # "MachIsentropic", "Load",
        "MassFlow"]

    parameterListN = [
        "PR", "TR",
        "Eff", "EffPoly",
        "PLoss", "Entr",
        "Mach", "Load",
        "MF"]

    work_path = os.path.join("..", "data", "data_collect", "surrogate_error")
    # Visual = MatplotlibVision(work_path, input_name=('Z', 'R'), field_name=parameterListN)  # 不在此处设置名称
    Visual = MatplotlibVision(work_path, input_name=('Z', 'R'), field_name=AlgoNameList)  # 不在此处设置名称
    fig_id = 0
    for ii, n_train in enumerate([2500]):#enumerate(n_trainList):
        for parameter in parameterList:
            Err_all = []
            for AlgoName in AlgoNameList:
                npz_path = os.path.join("..", "data", "surrogate_data", AlgoName + ".npz")
                data_pred = np.load(npz_path)

                value_true = data_true[parameter]
                value_true = value_true[-400:]
                value_true = value_true[:352]

                if AlgoName=="FNO":
                    idx = 0
                else:
                    idx = 4
                value_pred = data_pred[parameter][:, 0]
                value_pred = value_pred[:352]
                # 绘制ture-pred图
                # fig, axs = plt.subplots(1, 1, figsize=(10, 10), num=1)
                # title = AlgoName + '_' + parameter + '_' + str(n_train)
                # Visual.plot_regression(fig, axs, value_true.squeeze(), value_pred.squeeze(),
                #                        title= title)
                # jpg_path = os.path.join(work_path, title + '.jpg')
                # fig.savefig(jpg_path)
                # plt.close(fig)
                Err = np.abs((value_pred - value_true)/value_true)
                Err_all.append(Err[:,np.newaxis])

            # 绘制error-box
            title = parameter + '_' + str(n_train)
            Err_all = np.concatenate(Err_all, axis=1)
            fig, axs = plt.subplots(1, 1, figsize=(10, 10), num=1)
            Visual.plot_box(fig, axs, Err_all, xticks=Visual.field_name, title=parameter)

            jpg_path = os.path.join(work_path, title + "_algo_error-box.jpg")
            fig.savefig(jpg_path)
            plt.close(fig)


