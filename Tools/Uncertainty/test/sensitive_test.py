import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
torch.cuda.is_available()
import numpy as np
import matplotlib.pyplot as plt
from Tools.Uncertainty.SensitivityUncertainty import Turbo_UQLab
from Utilizes.visual_data import MatplotlibVision
from Tools.post_process.model_predict import predictor_establish
from utilizes_rotor37 import get_grid, get_origin_GVRB

if __name__ == "__main__":
    name = 'TNO_2'
    input_dim = 96
    output_dim = 8

    eval_dim = input_dim
    print(os.getcwd())
    work_load_path = os.path.join("../../../Demo/GVRB_2d", 'work')
    grid = get_grid(real_path='D:\WQN\CODE\DENO4pytorch-main\Demo\GVRB_2d\TestData', GV_RB=True)
    predictor = predictor_establish(name, work_load_path, predictor=True)
    predictor.grid_size_r = 128
    evaluater = lambda x:predictor.predicter_2d(x, input_norm=False)[..., :]
    # evaluater = lambda x: predictor.predicter_loader(x, input_norm=False)[..., 5]
    # evaluater = lambda x: predictor.predictor_cfd_value(x, input_norm=False)[..., 1]
    varlist = [15,15,15,3,15,15,15,3]
    startlist = [0] + np.cumsum(varlist)[:-1].tolist()
    for ii in range(8):
        var_dim = varlist[ii]
        var_start = startlist[ii]

        problem = {
            'num_vars': var_dim,  # 参数数量
            'names': [f'x{i+15}' for i in range(var_dim)],  # 参数名称
            'bounds': [[0, 1]] * var_dim,  # 参数范围
        }

        evaulate = {
            'input_shape': [96,],
            'names' : [f'x{i}' for i in range(eval_dim)],
            'output_shape': [128, 128, 1],
            'evaluate_func': evaluater,
            'reference': [0.5] * eval_dim,
        }

        # para = {
        #     'mu': 5,
        #     'sigma': 3,
        # }

        SA = Turbo_UQLab(problem, evaulate)
        data = SA.sample_generate(1280, dist='uniform',generate='lhs',paradict=None)
        print(data)
        rst = SA.value_evaluate(data)
        print(rst)
        mean = SA.moment_calculate(rst, type=1)
        var = np.sqrt(SA.moment_calculate(rst, type=2))

        Visual = MatplotlibVision('', input_name=('x', 'y'),
                                  field_name=('ps', 'ts', 'rho', 'vx', 'vy', 'vz', 'tt1', 'tt2'))
            #
        fig_id = ii
        fig, axs = plt.subplots(8, 2, figsize=(13, 25), num=2)
        Visual.plot_fields_ms_2col(fig, axs, mean[0], var[0], grid)
        fig.savefig(os.path.join(work_load_path, 'std_' + str(fig_id) + '.jpg'))
        plt.close(fig)

        #
        # plt.hist(rst, bins=30, density=True, alpha=0.6, color='b', label='Monte Carlo Samples')
        # plt.show()