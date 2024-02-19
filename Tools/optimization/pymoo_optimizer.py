import numpy as np
from pymoo.core.problem import Problem
import torch
import os
from Tools.post_process.model_predict import DLModelPost
from Tools.post_process.load_model import rebuild_model, build_model_yml
from Utilizes.process_data import DataNormer
from Tools.train_model.model_whole_life import WorkPrj
from Tools.uncertainty.GVRB_setting import InputTransformer, UQTransformer
# 定义目标函数
class TurboPredictor(Problem):
    def __init__(self,
                 model: DLModelPost=None,
                 adapter: InputTransformer=None,
                 # 软约束包含于parameterList, 硬约束不包含于parameterList
                 parameterList=None,
                 softconstrList=None,
                 hardConstrList=None,
                 hardConstrIneqList=None,
                 n_var: int=None,
                 is_uq_opt=False,
                ):

        self.model = model
        self.adapter = adapter
        self.parameterList = parameterList
        self.softconstrList = softconstrList
        self.hardConstrList = hardConstrList
        self.hardConstrIneqList = hardConstrIneqList
        n_obj = len(parameterList)
        if is_uq_opt:
            n_obj = len(adapter.uqList)

        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         n_constr=0,#len(hardConstrIneqList)+len(hardConstrList),
                         xl=0.0, xu=1.0)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.adapter.output_transformer(
                        self.model.predictor_cfd_value(
                        self.adapter.input_transformer(x),
                        parameterList=self.parameterList,
                        setOpt=True,
                        space=0,
                        ),
                        setOpt=True,
                        ) # 注意 这里修改过了。
        # 约束设置
        if self.hardConstrList is not None:
            if  len(self.hardConstrList) != 0:
                out["H"] = self.model.predictor_hardConstraint(self.adapter.input_transformer(x), hardconstrList=self.hardConstrList)
        if self.hardConstrIneqList is not None:
            if  len(self.hardConstrIneqList) != 0:
                out["G"] = self.model.predictor_hardConstraint(self.adapter.input_transformer(x), hardconstrList=self.hardConstrIneqList)


def predictor_establish(name, work_load_path):

    nameReal = name.split("_")[0]
    id = None
    if len(name.split("_")) == 2:
        id = int(name.split("_")[1])

    work_path = os.path.join(work_load_path, name)
    work = WorkPrj(work_path)
    Device = work.device
    print(Device)
    if os.path.exists(work.x_norm):
        norm_save_x = work.x_norm
        norm_save_y = work.y_norm
    else:
        assert False

    x_normlizer = DataNormer(np.ndarray([1, 1]), method="mean-std", axis=0)
    x_normlizer.load(norm_save_x)
    y_normlizer = DataNormer(np.ndarray([1, 1]), method="mean-std", axis=0)
    y_normlizer.load(norm_save_y)

    if os.path.exists(work.yml):
        Net_model, inference, _, _ = build_model_yml(work.yml, Device, name=nameReal)
        isExist = os.path.exists(work.pth)
        if isExist:
            checkpoint = torch.load(work.pth, map_location=Device)
            Net_model.load_state_dict(checkpoint['net_model'])
    else:
        Net_model, inference = rebuild_model(work_path, Device, name=nameReal)
    model_all = DLModelPost(Net_model, Device,
                        name=nameReal,
                        in_norm=x_normlizer,
                        out_norm=y_normlizer,
                        )
    return model_all

if __name__ == '__main__':
    var_name = ['S1_hub', 'ttem']
    test = InputTransformer(var_name)

    xx = np.random.random([100, test.num_var])
    xx_trans = test.input_transformer(xx)
    print(0)