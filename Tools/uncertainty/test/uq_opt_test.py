import numpy as np
from Tools.uncertainty.GVRB_setting import InputTransformer, UQTransformer

if __name__ == '__main__':
    var_name = ['S1_hub', 'S1_3d']
    uq_name = ['ttem']
    uq_number = 100
    # test = InputTransformer(var_name)
    test = UQTransformer(var_name, uq_name=uq_name, uq_number=uq_number)

    xx = np.random.random([100, test.num_var])
    xx_uq = test.input_transformer(xx)

    # xx_uq = test.output_transformer(test.input_transformer(xx))
    print(0)