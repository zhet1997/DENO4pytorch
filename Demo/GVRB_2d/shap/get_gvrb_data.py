import numpy as np
import os
import pyDOE
from Tools.optimization.pymoo_optimizer import TurboPredictor, predictor_establish, InputTransformer

if __name__ == "__main__":
    os.chdir(r'E:\\WQN\\CODE\\DENO4pytorch/')
    #定义设计空间
    name = 'TNO'
    work_load_path = os.path.join('Demo', 'GVRB_2d', 'work_opt')
    #var_name = ['S1','R1']
    var_name = ['S1_hub',
                'S1_pitch',
                'S1_tip',
                'S1_3d',
                'R1_hub',
                'R1_pitch',
                'R1_tip',]
    sample_num = 1000
    save_path = os.path.join(work_load_path, 'gvrb_data' + str(sample_num))

    #生成评估模型
    model_all = predictor_establish(name, work_load_path)
    adapter_gvrb = InputTransformer(var_name)
    problem = TurboPredictor(model=model_all,
                             adapter=adapter_gvrb,
                             n_var=adapter_gvrb.num_var,
                             parameterList=["Mass_flow",
                                            ],
                              )
    #生成评估样本
    np.random.seed(1)
    X = pyDOE.lhs(adapter_gvrb.num_var, samples=sample_num, criterion='maximin')
    #X[:,:48]=0.5
    #X=np.full(93,0.5)
    # 0, 63, 127
    dict_section = {'input':1,
                    'middle':59,
                    'output':127}
    Y = []
    #for key in dict_section.keys():
    Y.append(problem.direct_output(X, space=1, z1=0, z2=dict_section['output']))
    #保存评估样本
    dict_save = {}
    dict_save.update({'input': X})
    dict_save.update({'output': Y})

    if not os.path.exists(save_path): os.mkdir(save_path)
    np.savez(os.path.join(save_path, 'S1R1_Mass_1D.npz'), **dict_save)

