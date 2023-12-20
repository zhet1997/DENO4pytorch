import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
torch.cuda.is_available()
from Tools.post_process.model_predict import predictor_establish
from Tools.Uncertainty.SensitivityUncertainty import Turbo_UQLab
from draw_compare import *


def get_match_dict():
    varlist = [15, 15, 15, 3, 15, 15, 15, 3]
    startlist = [0] + np.cumsum(varlist)[:-1].tolist()
    var_idx = lambda id: range(startlist[id], varlist[id] + startlist[id])
    match_dict = {
        'S1_hub': var_idx(0),
        'S1_pitch': var_idx(1),
        'S1_tip': var_idx(2),
        'S1_3d': var_idx(3),
        'R1_hub': var_idx(4),
        'R1_pitch': var_idx(5),
        'R1_tip': var_idx(6),
        'R1_3d': var_idx(7),
        'tangle': [96],
        'ttem': [97],
        'tpre': [98],
        'rotate': [99],
    }
    return match_dict


def get_problem_set(name):
    match_dict = get_match_dict()
    range_dict = {
                'tangle': [-0.1, 0.1],
                'ttem': [699, 739],  # 719
                'tpre': [310000, 380000],  # 344740
                'rotate': [7500, 9100],  # 8279
                }

    problem = {
        'num_vars': len(match_dict[name]),  # 参数数量
        'names': [f'x{i}' for i in match_dict[name]],  # 参数名称
    }
    if name not in range_dict.keys():
        problem.update({'bounds': [[0, 1]] * len(match_dict[name])})
    else:
        problem.update({'bounds': [range_dict[name]]})

    return problem

def get_evaluate_set():
    evaulate = {
        'input_shape': [100, ],
        'names': [f'x{i}' for i in range(input_dim)],
        'output_shape': [64, 128, 1],
        # 'evaluate_func': evaluater,
        'reference': [0.5] * (input_dim - 4) + [0, 719, 344740, 8279],
    }
    return evaulate
def draw_sensitive_0d(predictor, work=None):
    evaluater = lambda x: predictor.predictor_cfd_value(x, input_norm=False, parameterList=parameterList, grid=grid,
                                                        space=0)
    match_dict = get_match_dict()
    # temp = [x for x in match_dict.keys() if not '_' in x]
    for var_name in match_dict.keys():
        ## get the draw data
        problem = get_problem_set(var_name)
        evaluate = get_evaluate_set()
        evaluate.update({'evaluate_func': evaluater})
        SA = Turbo_UQLab(problem, evaluate)
        data = SA.sample_generate(1280, dist='uniform', generate='lhs', paradict=None)
        rst = SA.value_evaluate(data)

        ## draw the figure
        save_path = os.path.join(work.root, 'save_figure')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        Visual = MatplotlibVision(os.path.join(work.root, 'save_figure'), input_name=('x', 'y'), field_name=('none'))

        for j in range(len(parameterList)):
            fig, axs = plt.subplots(1, 1, figsize=(10, 10))
            Visual.plot_histogram(fig, axs, rst[..., j], bins=50, range=None, color='navy', alpha=0.6)
            fig.savefig(os.path.join(save_path, 'hist_' + var_name + '_' + parameterList[j] + '.jpg'))
            plt.close(fig)

def draw_sensitive_1d(predictor, work=None):
    evaluater = lambda x: predictor.predictor_cfd_value(x, input_norm=False, parameterList=parameterList, grid=grid,
                                                        space=1)
    match_dict = get_match_dict()
    temp = [x for x in match_dict.keys() if 'S1_' in x]
    for var_name in temp:
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
        save_path = os.path.join(work.root, 'save_figure_test')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        Visual = MatplotlibVision(os.path.join(work.root, 'save_figure'), input_name=('x', 'y'), field_name=('none'))
        rangeList = draw_range_dict(parameterList)

        for j in range(len(parameterList)):
            fig, axs = plt.subplots(1, 1, figsize=(7, 9))
            Visual.plot_value_std_clean(fig, axs, mean[:, j], np.linspace(0, 1, mean.shape[0]),
                                        std=var[:, j], stdaxis=0, title=None, xlim=rangeList[j],
                                        xylabels=('x', 'y'), rangeIndex=5, color='darkgreen')
            fig.savefig(os.path.join(save_path, 'spanStd_' + var_name + '_' + parameterList[j].replace('/','') + '.jpg'))
            plt.close(fig)

def draw_sensitive_2d(predictor, work=None):
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


# draw figures include true and pred
if __name__ == '__main__':

    name = 'TNO_9'
    input_dim = 100
    output_dim = 8
    type = 'valid'
    var_name = 'S1_hub'
    parameterList = ['Static_pressure_ratio',
                     'Total_total_efficiency',
                     'Total_static_efficiency',
                     'Degree_reaction',
                     'atan(Vx/Vz)',
                     'atan(Wx/Wz)',
                     ]
    # parameterList = [
    #                  'Static Pressure',
    #                  # 'Relative Total Pressure',
    #                  'Absolute Total Pressure',
    #                  # 'Rotary Total Pressure',
    #                  'Static Temperature',
    #                  # 'Relative Total Temperature',
    #                  'Absolute Total Temperature',
    #                  # 'Rotary Total Temperature',
    #                  'Vx', 'Vy', 'Vz',
    #                  # '|V|','|V|^2',
    #                  'atan(Vx/Vz)',
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
    print(os.getcwd())
    work_load_path = os.path.join("../../Demo/GVRB_2d", 'work')
    work = WorkPrj(os.path.join(work_load_path, name))
    predictor = predictor_establish(name, work.root, predictor=True)
    grid = get_grid_interp(grid_num_s=64, grid_num_z=128)

    draw_sensitive_1d(predictor, work=work)









