import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
torch.cuda.is_available()
import numpy as np
from Tools.post_process.post_CFD import cfdPost_2d
from matplotlib import pyplot as plt
from Demo.GVRB_2d.utilizes_GVRB import get_origin

if __name__ == '__main__':
        design, fields, grids = get_origin(type='struct', realpath='..\data',
                                           quanlityList=["Static Pressure", "Static Temperature", "Density",
                                                         "Vx", "Vy", "Vz",
                                                         'Relative Total Temperature',
                                                         'Absolute Total Temperature'])
        inputdict = {'Static Pressure': 0,
                     'Static Temperature': 1,
                     'Density': 2,
                     'Vx': 3,
                     'Vy': 4,
                     'Vz': 5,
                     'Relative Total Temperature': 6,
                     'Absolute Total Temperature': 7,
                     }
        # variable_name_list = ['Total_total_efficiency', 'Total_static_efficiency', 'Static_pressure_ratio', 'Degree_reaction']
        post_true = cfdPost_2d(fields, grids, inputDict=inputdict)
        rst_TTEFF = post_true.get_performance('Total_total_efficiency', type='averaged')
        rst_TSEFF = post_true.get_performance('Total_static_efficiency', type='averaged')

        plt.scatter(np.arange(len(fields)), rst_TTEFF)
        plt.show()