from pymoo.optimize import minimize
import time
import os
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from Tools.optimization.pymoo_optimizer import TurboPredictor, predictor_establish
from Tools.uncertainty.GVRB_setting import UQTransformer, InputTransformer

if __name__ == "__main__":
    os.chdir(r'E:\\WQN\\CODE\\DENO4pytorch/')
    name = 'TNO'
    input_dim = 100
    output_dim = 5
    work_load_path = os.path.join('Demo', 'GVRB_2d', 'work_opt')
    var_name = ['S1']
    uq_name = ['ttem']
    uq_number = 5
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
    algorithm = NSGA2(pop_size=20)
    # algorithm = GA(pop_size=20)
    # 进行优化
    start_time = time.time()

    res = minimize(problem,
                   algorithm,
                   termination=('n_gen', 80),
                   verbose=True,
                   save_history=True
                   )  # 打印最优解

    end_time = time.time()

    print("最优解：", res.X)
    print("最优目标函数值：", res.F)
    print("运行时间：", end_time - start_time)

    # 保存到文件中

    # dict[parameter+"_sample"] = res.X
    # dict[parameter + "_value"] = res.F

        # np.savez(os.path.join("..", "data", "opt_data", 'sin_obj_minimize.npz'), **dict)

    # 保存数据
    # np.savez(os.path.join("..", "data", "opt_data", 'sin_obj_maximize.npz'), **dict)





    # n_evals = np.array([e.evaluator.n_eval for e in res.history])
    # opt = np.array([e.opt[0].F for e in res.history])
    # plt.scatter(opt[:, 0], opt[:, 1])
    # plt.show()
