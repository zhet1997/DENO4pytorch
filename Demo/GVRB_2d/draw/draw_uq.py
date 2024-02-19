import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
torch.cuda.is_available()
from Tools.draw_figure.draw_sensitive import *
from Tools.post_process.model_predict import DLModelPost
from Tools.uncertainty.GVRB_setting import InputTransformer, UQTransformer
from Tools.draw_figure.draw_opt_rst import optiaml_predicter

# draw figures include true and pred
if __name__ == '__main__':
    os.chdir(r'E:\\WQN\\CODE\\DENO4pytorch/')
    name = 'TNO'
    input_dim = 100
    output_dim = 8
    work_load_path = os.path.join('Demo', 'GVRB_2d', 'work_opt')
    var_name = 'S1'
    work = WorkPrj(os.path.join(work_load_path, name))
    predictor = predictor_establish(name, work.root, is_predictor=True)
    grid = get_grid_interp(grid_num_s=64, grid_num_z=128)

    parameterList = ['Static_pressure_ratio',
                     'Total_total_efficiency',
                     'Total_static_efficiency',
                     'Degree_reaction',
                     'atan(Vx/Vz)',
                     'atan(Wx/Wz)',
                     ]

    uqList = ['tangle', 'ttem', 'tpre', 'rotate']

    model_all = predictor_establish(name, work_load_path)
    adapter_gvrb = UQTransformer(var_name, uq_name=['tangle'], uq_number=1280)
    P = optiaml_predicter(model=model_all,
                          adapter=adapter_gvrb,
                          parameterList=parameterList,
                          )
    evaluater = lambda X:P.evaluate_with_bc_change(X, type='norm')

    draw_sensitive_0d(predictor, work=work, parameterList=parameterList)

