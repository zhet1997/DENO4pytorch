import numpy as np
import os
import pyDOE
from Tools.optimization.pymoo_optimizer import TurboPredictor, predictor_establish, InputTransformer
import torch.onnx

if __name__ == "__main__":
    os.chdir(r'E:\\WQN\\CODE\\DENO4pytorch/')
    #定义设计空间
    name = 'deepONet'
    work_load_path = os.path.join('Demo', 'GVRB_2d', 'work_opt')
    var_name = ['R1']
    sample_num = 500
    input_shape = (64, 128, 100,)
    coord_shape = (64, 128, 2,)
    save_path = os.path.join(work_load_path, 'gvrb_data' + str(sample_num))

    #生成评估模型
    model_all = predictor_establish(name, work_load_path, device_set='cpu')
    model = model_all.netmodel
    model.eval()

    x = (torch.randn(1, *input_shape), torch.randn(1, *coord_shape))  # 生成张量
    export_onnx_file = os.path.join(work_load_path, "don_test.onnx") # 目的ONNX文件名
    torch.onnx.export(model,
                      x,
                      export_onnx_file,
                      opset_version=10,
                      export_params=True,
                      do_constant_folding=True,  # 是否执行常量折叠优化
                      input_names=["input"],  # 输入名
                      output_names=["output"],  # 输出名
                      dynamic_axes={"input": {0: "batch_size"},  # 批处理变量
                                    "output": {0: "batch_size"}})