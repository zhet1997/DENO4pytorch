# compare |V^2| and |V|^2
# compare |V^2| and |V|^2


import numpy as np
import os
from Tools.post_process.post_CFD import cfdPost_2d
from scipy.interpolate import interp1d
# from utilizes_rotor37 import get_grid
import matplotlib.pyplot as plt
from Demo.GVRB_2d.utilizes_GVRB import get_grid, get_origin
from Demo.GVRB_2d.draw.draw_compare import *

if __name__ == "__main__":
    design, fields, grids = get_origin(
        type='struct',
        realpath=r'E:\\WQN\\CODE\\DENO4pytorch\\Demo\\GVRB_2d\\data',
        shuffled=True,
        getridbad=True,
        )

    post = cfdPost_2d(data=fields,
                      grid=grids,
                      boundarycondition=design[:,-4:],
                      similarity=False,
                      )
    rst1 = np.power(post.get_field('|V|^2'),0.5)
    rst2 = post.get_field('|V|')

    error = np.abs(rst1-rst2)

    Visual = MatplotlibVision('', input_name=('x', 'y'), field_name=('none'))
    Visual.field_name = 'V'
    i = 0
    fig, axs = plt.subplots(1, 2, figsize=(13, 30))
    Visual.plot_fields_ms_2col(fig, axs,
                               rst1[i,:,:,None],
                               error[i,:,:,None],
                               grids,
                               fmin_max=[[[0],[0]],[[500],[0.5]]]
                               )



    plt.show()
    print(0)

