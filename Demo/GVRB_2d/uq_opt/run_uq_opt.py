from pymoo.optimize import minimize
import time
import os
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from Tools.optimization.pymoo_optimizer import TurboPredictor, predictor_establish
from Tools.uncertainty.GVRB_setting import UQTransformer, InputTransformer

def get_history_value(history, name='X'):
    rst = []
    for population in history:
        pop_rst = []
        for data in population.pop:
            pop_rst.append(getattr(data, name))
        rst.append(pop_rst)
    return rst

if __name__ == "__main__":
    os.chdir(r'E:\\WQN\\CODE\\DENO4pytorch/')
    name = 'TNO'
    input_dim = 100
    output_dim = 5
    work_load_path = os.path.join('Demo', 'GVRB_2d', 'work_opt')
    var_name = ['R1']
    uq_name = ['tangle',
        'ttem',
        'tpre',
        'rotate',]
    uq_number = 5000
    model_all = predictor_establish(name, work_load_path)
    adapter_gvrb = UQTransformer(var_name, uq_name=uq_name, uq_number=uq_number)
    problem = TurboPredictor(model=model_all,
                             adapter=adapter_gvrb,
                             is_uq_opt=True,
                             n_var=adapter_gvrb.num_var,
                             parameterList=["Total_total_efficiency"],
                            # softconstrList=[],
                            # hardConstrList=["MassFlow"],
                            # hardConstrIneqList=[],
                              )
    # 定义优化算法
    algorithm = NSGA2(pop_size=32)
    # algorithm = GA(pop_size=20)
    # 进行优化
    start_time = time.time()

    res = minimize(problem,
                   algorithm,
                   termination=('n_gen', 100),
                   verbose=True,
                   save_history=True
                   )  # 打印最优解

    end_time = time.time()
    print("最优解：", res.X)
    print("最优目标函数值：", res.F)
    print("运行时间：", end_time - start_time)

    dict_rst = {}
    for name in ('X', 'F'):
        dict_rst.update({name: np.array(get_history_value(res.history, name))})
    dict_rst.update({'time': end_time - start_time})
    save_path = os.path.join(work_load_path, 'opt_rst_R1')
    if not os.path.exists(save_path) : os.mkdir(save_path)
    np.savez(os.path.join(save_path, 'uq_opt.npz'), **dict_rst)

