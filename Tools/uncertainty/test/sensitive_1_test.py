import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
torch.cuda.is_available()
import numpy as np
import matplotlib.pyplot as plt
from Tools.uncertainty.SensitivityUncertainty import Turbo_UQLab
from Utilizes.visual_data import MatplotlibVision
from Tools.post_process.model_predict import predictor_establish
from utilizes_rotor37 import get_grid, get_origin_GVRB

if __name__ == "__main__":
    name = 'Transformer'
    input_dim = 96
    output_dim = 8
    var_dim = 15
    eval_dim = input_dim
    print(os.getcwd())
    work_load_path = os.path.join("../..", 'work_Trans5000_2')
    grid = get_grid(real_path='D:\WQN\CODE\DENO4pytorch-main\Demo\GV_RB\TestData', GV_RB=True)
    predictor = predictor_establish(name, work_load_path, predictor=True)
    evaluater = lambda x:predictor.predicter_2d(x, input_norm=False)
    problem = {
        'num_vars': var_dim,  # 参数数量
        'names': [f'x{i}' for i in range(var_dim)],  # 参数名称
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
    data = SA.sample_generate(128, dist='uniform',generate='lhs',paradict=None)
    print(data)
    rst = SA.value_evaluate(data)
    print(rst)
    sa = SA.moment_calculate(rst, type=1)
    print(sa)
    # plt.hist(rst, bins=30, density=True, alpha=0.6, color='b', label='Monte Carlo Samples')
    # plt.show()