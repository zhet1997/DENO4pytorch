import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import numpy as np
torch.cuda.is_available()
from Tools.draw_figure.draw_uncetainty import draw_uq_0d, draw_uq_0d_multi_bc, draw_uq_1d
from Tools.optimization.pymoo_optimizer import predictor_establish
from Tools.post_process.model_predict import DLModelPost
from Tools.uncertainty.GVRB_setting import InputTransformer, UQTransformer
from Tools.draw_figure.draw_opt_rst import optiaml_predicter
import matplotlib.pyplot as plt
from draw_opt import get_compare_data, load_opt_data

def draw_uq_list():
    for uq in uqList:
        adapter_gvrb = UQTransformer(var_name, uq_name=uq, uq_number=10000)
        P = optiaml_predicter(model=model_all,
                              adapter=adapter_gvrb,
                              parameterList=parameterList,
                              )
        evaluater = lambda X:P.evaluate_with_bc_change(X, type='norm')


        draw_uq_0d(X,
                   evaluater=evaluater,
                   save_path=os.path.join(work_load_path,'save_figure'),
                   parameterList=parameterList,
                   uq_name=uq,
                   )

def draw_uq_all():
    adapter_gvrb = UQTransformer(var_name, uq_name=uqList, uq_number=1000)
    P = optiaml_predicter(model=model_all,
                          adapter=adapter_gvrb,
                          parameterList=parameterList,
                          )
    evaluater = lambda X: P.evaluate_with_bc_change(X, type='norm')

    draw_uq_0d(X,
               evaluater=evaluater,
               save_path=os.path.join(work_load_path, 'save_figure'),
               parameterList=parameterList,
               uq_name='All',
               colorList=list(colorList),
               )

# draw figures include true and pred
if __name__ == '__main__':
    os.chdir(r'E:\\WQN\\CODE\\DENO4pytorch/')
    name = 'TNO'
    input_dim = 100
    output_dim = 8
    work_load_path = os.path.join('Demo', 'GVRB_2d', 'work_opt')
    var_name = ['S1','R1']
    parameterList = [
                     # 'Static_pressure_ratio',
                     'Total_total_efficiency',
                     # 'Total_static_efficiency',
                     'Degree_reaction',
                     'Mass_flow'
                     # 'atan(Vx/Vz)',
                     # 'atan(Wx/Wz)',
                     ]

    uqList = ['tangle', 'ttem', 'tpre', 'rotate']
    cmap = plt.cm.get_cmap('tab10')
    colorList = cmap.colors

    # X = np.array([0.5] * 48)

    X = get_compare_data(var_name)

    # X, _, _ = load_opt_data(var_name, type='base')

    model_all = predictor_establish(name, work_load_path)
    draw_uq_all()



    # draw_uq_0d_multi_bc(X, model_all=model_all,
    #                     save_path=os.path.join(work_load_path, 'save_figure'),
    #                     parameterList=parameterList, uqList=uqList, colorList=colorList, var_name=var_name)

    #
    # adapter_gvrb = UQTransformer(var_name, uq_name=uqList, uq_number=1000)
    # P = optiaml_predicter(model=model_all,
    #                       adapter=adapter_gvrb,
    #                       parameterList=parameterList,
    #                       )
    # evaluater = lambda X:P.evaluate_with_bc_change(X, type='norm', space=1)
    # draw_uq_1d(X,
    #            evaluater=evaluater,
    #            save_path=os.path.join(work_load_path,'save_figure_1d'),
    #            parameterList=parameterList,
    #            uq_name=uqList,
    #            )



