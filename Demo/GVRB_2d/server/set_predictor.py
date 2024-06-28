import numpy as np
import os
import pyDOE
from Tools.optimization.pymoo_optimizer import TurboPredictor, predictor_establish, InputTransformer


def build_predicter():
    os.chdir(r'E:\\WQN\\CODE\\DENO4pytorch/')
    # 定义设计空间
    name = 'TNO'
    work_load_path = os.path.join('Demo', 'GVRB_2d', 'work_opt')
    model_all = predictor_establish(name, work_load_path)

    return model_all

def run_predicter(X, model=None, parameter=("Mass_flow",)):

    var_name = ['S1_hub',
                'S1_pitch',
                'S1_tip',
                'S1_3d',
                'R1_hub',
                'R1_pitch',
                'R1_tip', ]
    # 生成评估模型
    adapter_gvrb = InputTransformer(var_name)
    problem = TurboPredictor(model=model,
                             adapter=adapter_gvrb,
                             n_var=adapter_gvrb.num_var,
                             parameterList=parameter,
                             )
    Y=problem.direct_output(X, space=0)
    return Y

def s2n(s):
    s = s.replace('*enter*', '\n')
    s = s.strip('\n')
    data = []
    for letter in s.split():
        data.append(float(letter))
    return data

def n2s(x):
    data = ''
    for num in x:
        data = data + '\t' + str(num)
    data = data + '*enter*'
    return data




if __name__ == "__main__":
    X = pyDOE.lhs(93, samples=32, criterion='maximin')
    model = build_predicter()


    for ii in range(32):
        data = n2s(X[ii])
        data = np.array(s2n(data))

        Y = run_predicter(data, model=model)
        print(Y)


