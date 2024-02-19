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

def get_uq_0d(data, evaluater=None,
              expand=100,
               **kwargs
               ):
    var_name = 'S1_hub'
    problem = get_problem_set(var_name)
    evaluate = get_evaluate_set()
    evaluate.update({'evaluate_func': evaluater})
    SA = Turbo_UQLab(problem, evaluate)
    # data = SA.sample_generate(1280, dist='uniform', generate='lhs', paradict=None)
    rst = SA.value_pure_evaluate(data)*100

    return rst

def draw_uq_0d(data, evaluater=None,
               save_path=None,
               parameterList=None,
               uq_name=None,
               **kwargs
               ):
    # match_dict = get_match_dict()
    # temp = [x for x in match_dict.keys() if not '_' in x]
    # for var_name in match_dict.keys():
        ## get the draw data
    # var_name = 'S1_hub'
    # problem = get_problem_set(var_name)
    # var_name = 'S1'
    # evaluate = get_evaluate_set()
    # evaluate.update({'evaluate_func': evaluater})
    # SA = Turbo_UQLab(problem, evaluate)
    # # data = SA.sample_generate(1280, dist='uniform', generate='lhs', paradict=None)
    # rst = SA.value_pure_evaluate(data)
    rst = get_uq_0d(data, evaluater=evaluater)

    ## draw the figure
    if save_path is None:
        save_path = os.path.join(work.root, 'save_figure')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    Visual = MatplotlibVision('', input_name=('x', 'y'), field_name=('none'))


    for j in range(len(parameterList)):
        fig, axs = plt.subplots(1, 1, figsize=(10, 10))
        Visual.plot_histogram(fig, axs, rst[..., j], bins=20, range=None, color='navy', alpha=0.6)
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
        save_path = os.path.join(work.root, 'save_figure')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    Visual = MatplotlibVision('', input_name=('x', 'y'), field_name=('none'))


    for j in range(len(parameterList)):
        range_min = np.nan
        range_max = np.nan
        fig, axs = plt.subplots(1, 1, figsize=(10, 10))
        data_draw = []
        for k, uq in enumerate(uqList):
            range_min = np.nanmin(np.append(rst_dict[uq][..., j], range_min))
            range_max = np.nanmax(np.append(rst_dict[uq][..., j], range_max))
            data_draw.append(rst_dict[uq][..., j])
        Visual.plot_histogram(fig, axs, data_draw, bins=30, rangeList=[range_min, range_max], color=colorList[:4],
                              alpha=0.5, label=uqList)
        # for k, uq in enumerate(uqList):
        #     Visual.plot_histogram(fig, axs, rst_dict[uq][..., j], bins=30, range=[range_min, range_max], color=colorList[k], alpha=0.5, label=uq)
        fig.savefig(os.path.join(save_path, 'hist_' + 'all_' + var_name + '_' + parameterList[j] + '.jpg'))
        plt.close(fig)
def draw_uq_1d(predictor, work=None, parameterList=None, grid=None):
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


# draw figures include true and pred
if __name__ == '__main__':

    name = 'TNO_5'
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
    work_load_path = os.path.join("E:\WQN\CODE\DENO4pytorch\Demo\GVRB_2d", 'work_opt')
    work = WorkPrj(os.path.join(work_load_path, name))
    predictor = predictor_establish(name, work.root, is_predictor=True)
    grid = get_grid_interp(grid_num_s=64, grid_num_z=128)

    draw_sensitive_1d(predictor, work=work)









