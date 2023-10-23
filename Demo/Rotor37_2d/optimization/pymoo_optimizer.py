from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
import time
import paddle
import os
from post_process.model_predict import predictor_establish

# 定义目标函数
class Rotor37Predictor(Problem):

    def __init__(self, model,  # 软约束包含于parameterList, 硬约束不包含于parameterList
                 parameterList=None,
                 softconstrList=None,
                 hardConstrList=None,
                 hardConstrIneqList=None,
                ):

        self.model = model
        self.parameterList = parameterList
        self.softconstrList = softconstrList
        self.hardConstrList = hardConstrList
        self.hardConstrIneqList = hardConstrIneqList

        super().__init__(n_var=28,
                         n_obj=len(parameterList),
                         n_constr=0,#len(hardConstrIneqList)+len(hardConstrList),
                         xl=0.0, xu=1.0)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.model.predictor_value(x, parameterList=self.parameterList, setOpt=True) # 注意 这里修改过了。
        # 约束设置
        if self.hardConstrList is not None:
            if  len(self.hardConstrList) != 0:
                out["H"] = self.model.predictor_hardConstraint(x, hardconstrList=self.hardConstrList)
        if self.hardConstrIneqList is not None:
            if  len(self.hardConstrIneqList) != 0:
                out["G"] = self.model.predictor_hardConstraint(x, hardconstrList=self.hardConstrIneqList)


if __name__ == "__main__":
    # 设置需要优化的函数
    name = 'FNO'
    input_dim = 28
    output_dim = 5
    work_load_path = os.path.join("..", 'work', 'model_save')
    # work_load_path = os.path.join("..", "work")

    model_all = predictor_establish(name, work_load_path)

    parameterList = [
        "Efficiency",
        "EfficiencyPoly",
        "PressureRatioV",
        "TemperatureRatioV",
        "PressureRatioW",
        "TemperatureRatioW",
        "PressureLossR",
        "EntropyStatic",
        "MachIsentropic",
        "Load",
        "MassFlow"
    ]

    # 单个对象优化
    dict = {}
    # for parameter in parameterList:
        # 创建问题对象
    problem = Rotor37Predictor(model_all,
                               parameterList=["Efficiency"],
                               # softconstrList=[],
                               # hardConstrList=["MassFlow"],
                               # hardConstrIneqList=[],
                              )
    # 定义优化算法
    # algorithm = NSGA2(pop_size=10)
    algorithm = GA(pop_size=20)
    # 进行优化
    start_time = time.time()

    res = minimize(problem,
                   algorithm,
                   termination=('n_gen', 80),
                   verbose=True,
                   # save_history=True
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
