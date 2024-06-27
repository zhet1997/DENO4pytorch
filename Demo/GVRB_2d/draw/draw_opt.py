import os
import matplotlib.pyplot as plt
import numpy as np
from Tools.draw_figure.draw_opt_rst import simple_cull, optiaml_predicter, draw_performance_with_bc_change
from Tools.optimization.pymoo_optimizer import TurboPredictor, predictor_establish
from Tools.uncertainty.GVRB_setting import UQTransformer, InputTransformer
from Tools.draw_figure.draw_uncetainty import get_uq_0d

def load_opt_data(var_name, type):
    file_name_dict = {
        'norm' : 'norm_opt.npz',
        'uq' : 'uq_opt.npz',
        'uq_ts': 'uq_opt.npz'
    }

    file_path_dict = {
        'norm': 'opt_norm_rst',
        'uq': 'opt_rst',
        'uq_ts': 'opt_rst_ts'
    }

    if type=='base':
        X = np.array([0.5] * 48 * len(var_name))[np.newaxis,:]
        F = None
        save_path = None
    else:
        if len(var_name)==2:
            var_name = 'S1R1'
        elif len(var_name)==0:
            var_name = 'base'
        else:
            var_name = var_name[0]

        work_load_path = os.path.join('Demo', 'GVRB_2d', 'work_opt')
        save_path = os.path.join(work_load_path, file_path_dict[type] + '_' + var_name)
        file_name = os.path.join(save_path, file_name_dict[type])
        data = np.load(file_name)

        F = data['F'].reshape([-1, data['F'].shape[-1]])
        X = data['X'].reshape([-1, data['X'].shape[-1]])

    return X, F, save_path
def draw_pareto():
    os.chdir(r'E:\\WQN\\CODE\\DENO4pytorch/')
    name = 'TNO'
    work_load_path = os.path.join('Demo', 'GVRB_2d', 'work_opt')
    save_path = os.path.join(work_load_path, 'save_20240627')

    if not os.path.exists(save_path): os.mkdir(save_path)
    var_name = ['S1', 'R1']
    model_all = predictor_establish(name, work_load_path)


    X, F, _ = load_opt_data(var_name, type='uq_ts')
    # dict_data = get_compare_data(var_name, return_dict=True)

    paretoPoints, dominatedPoints, paretoIdx = simple_cull(F[-100:].tolist())
    print(len(paretoIdx))
    uqList = ['tangle', 'ttem', 'tpre', 'rotate']
    adapter_gvrb = UQTransformer(var_name, uq_name=uqList, uq_number=1280)
    P = optiaml_predicter(model=model_all,
                          adapter=adapter_gvrb,
                          parameterList=['Total_total_efficiency'],
                          )
    evaluater = lambda X: P.evaluate_with_bc_change(X, type='norm')
    F_uq = get_uq_0d(X[paretoIdx], evaluater=evaluater)
    F_uq = adapter_gvrb.output_transformer(F_uq, setOpt=False)
    # plt.scatter(dominatedPoints[:, 0], dominatedPoints[:, 1], s=10, alpha=0.5)

    # plt.scatter(dominatedPoints[:, 0], dominatedPoints[:, 1], s=10, alpha=0.5)

    plt.scatter(F_uq[:, 0], F_uq[:, 1], s=20, alpha=0.5, c='k')
    plt.scatter(paretoPoints[:, 0]*-1, paretoPoints[:, 1], s=20, alpha=0.5)


    plt.savefig(os.path.join(save_path, 'pareto.png'), dpi=600)
    plt.show()
    plt.close()

    # base_X = X[0:1, :]  # the first sample
    # opt_X = X[-2:-1, :]  # the last sample
    #
    # uq_name = ['ttem', ]
    # uq_number = 100
    # adapter_gvrb = UQTransformer(var_name, uq_name=uq_name, uq_number=uq_number)
    #
    # P = optiaml_predicter(model=model_all,
    #                       adapter=adapter_gvrb,
    #                       parameterList=['Total_total_efficiency', 'Degree_reaction'],
    #                       )
    #
    # base_F_bc = P.evaluate_with_bc_change(base_X)
    # opt_F_bc = P.evaluate_with_bc_change(opt_X)
    #
    # plt.plot(np.linspace(0, 1, uq_number), base_F_bc[:, 0])
    # plt.plot(np.linspace(0, 1, uq_number), opt_F_bc[:, 0])
    #
    # plt.show()
    # print(0)


def draw_bc_curves():
    os.chdir(r'E:\\WQN\\CODE\\DENO4pytorch/')
    name = 'TNO'
    input_dim = 100
    output_dim = 5
    work_load_path = os.path.join('Demo', 'GVRB_2d', 'work_opt')
    var_name = ['S1', 'R1']
    model_all = predictor_establish(name, work_load_path)

    X, F, save_path = load_opt_data(var_name, type='uq')
    X_norm, F_norm , _= load_opt_data(var_name, type='norm')
    X_base, _, _ = load_opt_data(var_name, type='base')


    # draw_performance_with_bc_change(X[::10], var_name=var_name, uq_number=100, work_load_path=work_load_path,
    #                                 save_path=save_path,
    #                                 parameterList=['Total_total_efficiency','Total_static_efficiency','Degree_reaction','Mass_flow'],
    #                                 uqList=['tangle','ttem','tpre','rotate'],
    #                                 )

    # draw_performance_with_bc_change(X_norm[::40], var_name=var_name, uq_number=100, work_load_path=work_load_path,
    #                                 save_path=save_path_norm,
    #                                 parameterList=['Total_total_efficiency', 'Total_static_efficiency',
    #                                                'Degree_reaction', 'Mass_flow'],
    #                                 uqList=['tangle', 'ttem', 'tpre', 'rotate'],
    #                                 )
    compredata = np.concatenate([X_base, X_norm[-1301:-1300], X[-33:-1]])
    draw_performance_with_bc_change(compredata, var_name=var_name, uq_number=100, work_load_path=work_load_path,
                                    save_path=save_path,
                                    parameterList=['Total_total_efficiency', 'Total_static_efficiency',
                                                   'atan(Vx/Vz)',
                                                   'Degree_reaction', 'Mass_flow'],
                                    uqList=['tangle', 'ttem', 'tpre', 'rotate'],
                                    )

def get_compare_data(var_name, return_dict=False):
    X, F, _ = load_opt_data(var_name, type='uq')
    X_norm, F_norm, _ = load_opt_data(var_name, type='norm')
    X_base, _, _ = load_opt_data(var_name, type='base')

    if return_dict:
        compre_dict = {}
        compre_dict.update({'base': X_base})
        compre_dict.update({'uq': X})
        compre_dict.update({'dtm': X_norm})
        return compre_dict
    else:
        compredata = np.concatenate([X_base, X_norm[800:801], X[-5:-1]])
        return compredata


if __name__ == "__main__":
    draw_pareto()
