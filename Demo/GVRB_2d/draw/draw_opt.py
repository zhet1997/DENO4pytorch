import os
import matplotlib.pyplot as plt
import numpy as np
from Tools.draw_figure.draw_opt_rst import simple_cull, optiaml_predicter, draw_performance_with_bc_change
from Tools.optimization.pymoo_optimizer import TurboPredictor, predictor_establish
from Tools.uncertainty.GVRB_setting import UQTransformer, InputTransformer

def draw_pareto():
    os.chdir(r'E:\\WQN\\CODE\\DENO4pytorch/')
    name = 'TNO'
    input_dim = 100
    output_dim = 5
    work_load_path = os.path.join('Demo', 'GVRB_2d', 'work_opt')
    var_name = ['S1']
    model_all = predictor_establish(name, work_load_path)

    save_path = 'E:\WQN\CODE\DENO4pytorch\Demo\GVRB_2d\work_opt\opt_rst_3/'
    file_name = os.path.join(save_path, 'uq_opt.npz')
    data = np.load(file_name)
    print(0)
    F = data['F'].reshape([-1, data['F'].shape[-1]])

    paretoPoints, dominatedPoints = simple_cull(F.tolist())

    plt.scatter(dominatedPoints[:, 0], dominatedPoints[:, 1], s=10, alpha=0.5)
    plt.scatter(paretoPoints[:, 0], paretoPoints[:, 1], s=20, alpha=0.5)
    plt.savefig(save_path + 'pareto.png', dpi=600)
    plt.close()

    X = data['X'].reshape([-1, data['X'].shape[-1]])
    base_X = X[0:1, :]  # the first sample
    opt_X = X[-2:-1, :]  # the last sample

    uq_name = ['ttem', ]
    uq_number = 100
    adapter_gvrb = UQTransformer(var_name, uq_name=uq_name, uq_number=uq_number)

    P = optiaml_predicter(model=model_all,
                          adapter=adapter_gvrb,
                          parameterList=['Total_total_efficiency', 'Degree_reaction'],
                          )

    base_F_bc = P.evaluate_with_bc_change(base_X)
    opt_F_bc = P.evaluate_with_bc_change(opt_X)

    plt.plot(np.linspace(0, 1, uq_number), base_F_bc[:, 0])
    plt.plot(np.linspace(0, 1, uq_number), opt_F_bc[:, 0])

    plt.show()
    print(0)


if __name__ == "__main__":
    os.chdir(r'E:\\WQN\\CODE\\DENO4pytorch/')
    name = 'TNO'
    input_dim = 100
    output_dim = 5
    work_load_path = os.path.join('Demo', 'GVRB_2d', 'work_opt')
    var_name = ['S1']
    model_all = predictor_establish(name, work_load_path)

    save_path = 'E:\WQN\CODE\DENO4pytorch\Demo\GVRB_2d\work_opt\opt_rst_3/'
    file_name = os.path.join(save_path, 'uq_opt.npz')
    data = np.load(file_name)
    F = data['F'].reshape([-1, data['F'].shape[-1]])
    X = data['X'].reshape([-1, data['X'].shape[-1]])
    draw_performance_with_bc_change(X[::10], var_name=var_name, uq_number=100, work_load_path=work_load_path,
                                    save_path=save_path,
                                    parameterList=['Total_total_efficiency','Total_static_efficiency','Degree_reaction','Mass_flow'],
                                    uqList=['tangle','ttem','tpre','rotate'],
                                    )