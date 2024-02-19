from pymoo.optimize import minimize
import time
import numpy as np
import os
from pymoo.algorithms.soo.nonconvex.ga import GA
from Tools.optimization.pymoo_optimizer import TurboPredictor, predictor_establish, InputTransformer
from run_uq_opt import get_history_value


if __name__ == "__main__":
    os.chdir(r'E:\\WQN\\CODE\\DENO4pytorch/')
    name = 'TNO'
    input_dim = 100
    output_dim = 5
    work_load_path = os.path.join('Demo', 'GVRB_2d', 'work_opt')
    var_name = ['S1','R1']
    model_all = predictor_establish(name, work_load_path)
    adapter_gvrb = InputTransformer(var_name)

    # 单个对象优化
    dict = {}
    problem = TurboPredictor(model=model_all,
                             adapter=adapter_gvrb,
                             n_var=adapter_gvrb.num_var,
                             parameterList=["Total_total_efficiency"],
                            # softconstrList=[],
                            # hardConstrList=["MassFlow"],
                            # hardConstrIneqList=[],
                              )
    # 定义优化算法
    # algorithm = NSGA2(pop_size=10)
    algorithm = GA(pop_size=32)
    # 进行优化
    start_time = time.time()

    res = minimize(problem,
                   algorithm,
                   termination=('n_gen', 100),
                   verbose=True,
                   save_history=True,
                   )  # 打印最优解

    end_time = time.time()

    print("最优解：", res.X)
    print("最优目标函数值：", res.F)
    print("运行时间：", end_time - start_time)

    dict_rst = {}
    for name in ('X', 'F'):
        dict_rst.update({name: np.array(get_history_value(res.history, name))})
    dict_rst.update({'time': end_time - start_time})
    save_path = os.path.join(work_load_path, 'opt_norm_rst_S1R1')
    if not os.path.exists(save_path): os.mkdir(save_path)
    np.savez(os.path.join(save_path, 'norm_opt.npz'), **dict_rst)
