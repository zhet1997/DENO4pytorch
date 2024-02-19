import numpy as np
from Tools.uncertainty.SensitivityUncertainty import Turbo_UQLab
def apply_opt(func):
    def wrapper(instance, *args, **kwargs):
        rst = func(instance, *args, **kwargs)
        if kwargs['setOpt']:
            for ii, parameter_Name in enumerate(instance.uqList):
                rst[:,ii] = rst[:,ii] * MaxOrMIn_uq(parameter_Name)
        return rst
    return wrapper
class InputTransformer(object):
    def __init__(self, var_name):
        self.evaluate = get_evaluate_set()
        self.problem = get_problem_set(var_name, expand=True)

        self.num_var = self.problem['num_vars']
        self.names_input = self.evaluate['names']
        self.names_var = self.problem['names']
        self.range_var = self.problem['bounds']
        self.reference = self.evaluate['reference']

        self.opt_slice_index = self.get_problem_slice(self.problem)
        self.uqList = None
    def get_slice(self):
        slice_index = []
        for ii in range(self.num_var):
            slice_index.append(self.names_input.index(self.names_var[ii]))
        self.slice_index = slice_index

    def get_problem_slice(self, problem):
        slice_index = []
        for ii in range(problem['num_vars']):
            slice_index.append(self.names_input.index(problem['names'][ii]))
        return slice_index

    def input_transformer(self, x_var, is_norm=True, **kwargs):
        assert self.num_var==x_var.shape[-1]
        if is_norm:
            range = np.array(self.range_var)
            x_var = x_var * (range[np.newaxis, :,1] - range[np.newaxis, :,0]) + range[np.newaxis, :,0]
        x_input = np.tile(self.reference, [x_var.shape[0],1])
        x_input[:, self.opt_slice_index] = x_var

        return x_input

    def output_transformer(self, output, **kwargs):
        return output



class UQTransformer(InputTransformer):
    def __init__(self, var_name, uq_name=None, uq_number=None, uqList=None):
        super(UQTransformer, self).__init__(var_name)
        self.uq_number = uq_number
        uq_problem = get_problem_set(uq_name, expand=True)
        self.uq_input = Turbo_UQLab(uq_problem, self.evaluate)
        self.uq_slice_index = self.get_problem_slice(uq_problem)
        self.uq_problem = uq_problem

        if uqList is None:
            uqList = ['mean','var']
        self.uqList = uqList

    def get_monte_carlo_group(self, number, type='norm'):
        if type=='norm':
            data = self.uq_input.sample_generate(number, dist='norm', generate='random', paradict={'mu':0.5, 'sigma':0.1})
        elif type=='uniform':
            data = self.uq_input.sample_generate(number, dist='uniform', generate='lhs', paradict=None)
        elif type=='linear':
            data = np.linspace(0.2,0.8, number)
        else:
            assert False
        if len(data.shape)==1:
            data = data[..., np.newaxis]
        return data

    def get_bc_change_group(self, x_var, is_norm=True, bc_number=1):
        uq_input = self.get_monte_carlo_group(number=bc_number, type='linear')
        if is_norm:
            range = np.array(self.uq_problem['bounds'])
            uq_input = uq_input * (range[np.newaxis, :, 1] - range[np.newaxis, :, 0]) + range[np.newaxis, :, 0]
        return uq_input

    def input_transformer(self, x_var, is_norm=True, type='norm',perturb=True):
        x_input = super().input_transformer(x_var, is_norm=is_norm)
        uq_input = self.get_monte_carlo_group(number=self.uq_number, type=type)
        if is_norm:
            range = np.array(self.uq_problem['bounds'])
            uq_input = uq_input * (range[np.newaxis, :,1] - range[np.newaxis, :,0]) + range[np.newaxis, :,0]
        x_input = np.tile(x_input, [uq_input.shape[0], 1])


        if perturb:
            set_problem = get_problem_set( ['tangle', 'ttem', 'tpre', 'rotate'], expand=True)
            range_p = np.repeat(np.array(set_problem['bounds'])[np.newaxis,...], x_input.shape[0], axis=0)
            perturb_input = np.random.normal(loc=0.5, scale=0.02, size=[x_input.shape[0],4])
            perturb_input = perturb_input * (range_p[:, :, 1] - range_p[:, :, 0]) + range_p[:, :, 0]
            x_input[:, -4:] = perturb_input

        x_input[:, self.uq_slice_index] = np.repeat(uq_input, x_var.shape[0], axis=0)

        return x_input

    @apply_opt
    def output_transformer(self, output, setOpt=True):
        rst = []
        output = output.reshape([self.uq_number, -1, output.shape[-1]])  #
        for uq in self.uqList:
            rst.append(self.uq_input.moment_calculate(output, type=uq, opera_axis=0, squeeze=True))
        return np.concatenate(rst, axis=1)

def get_match_dict():
    varlist = [15, 15, 15, 3, 15, 15, 15, 3]
    startlist = [0] + np.cumsum(varlist)[:-1].tolist()
    var_idx = lambda id: range(startlist[id], varlist[id] + startlist[id])
    match_dict = {
        'S1_hub': var_idx(0),
        'S1_pitch': var_idx(1),
        'S1_tip': var_idx(2),
        'S1_3d': var_idx(3),
        'R1_hub': var_idx(4),
        'R1_pitch': var_idx(5),
        'R1_tip': var_idx(6),
        'R1_3d': var_idx(7),
        'tangle': [96],
        'ttem': [97],
        'tpre': [98],
        'rotate': [99],
        'default': None,
    }
    return match_dict

def get_range_dict():
    # range_dict = {
    #     'tangle': [-0.1, 0.1],
    #     'ttem': [699, 739],  # 719
    #     'tpre': [310000, 380000],  # 344740
    #     'rotate': [7500, 9100],  # 8279
    #     'default': [0, 1],
    # }
    range_dict = {
        'x96': [-0.1, 0.1],
        'x97': [699, 739],  # 719
        'x98': [314740, 374740],  # 344740
        'x99': [7979, 8579],  # 8279
        'default': [0.3, 0.7],
    }
    return range_dict


def get_problem_set(name, expand=False):
    match_dict = get_match_dict()
    if not isinstance(name, list):
        name_list = [name, ]
    else:
        name_list = name

    if expand:
        temp = []
        for nn in name_list:
            temp.extend([x for x in match_dict.keys() if nn in x])
        name_list = temp

    range_dict = get_range_dict()
    var_idx = get_list_from_dict(name_list, match_dict)

    problem = {
        'num_vars': len(var_idx),  # 参数数量
        'names': [f'x{i}' for i in var_idx],  # 参数名称
    }

    problem.update({'bounds': get_data_from_dict(problem['names'], range_dict)})

    return problem

def get_list_from_dict(name_list: list, dict: dict):
    rst = []
    for name in name_list:
        if name in dict.keys():
            rst.extend(list(dict[name]))
        else:
            rst.extend(list(dict['default']))
    return rst
def get_data_from_dict(name_list: list, dict: dict):
    rst = []
    for name in name_list:
        if name in dict.keys():
            rst.append(dict[name])
        else:
            rst.append(dict['default'])

    return rst

def get_evaluate_set(input_dim=100):
    evaulate = {
        'input_shape': [100, ],
        'names': [f'x{i}' for i in range(input_dim)],
        'output_shape': [64, 128, 1],
        # 'evaluate_func': evaluater,
        'reference': [0.5] * (input_dim - 4) + [0, 719, 344740, 8279],
    }
    return evaulate


def MaxOrMIn_gvrb(parameter):
    dict = {
    'Static_pressure_ratio': -1,
    'Absolute_total_pressure_ratio': -1,
    'Absolute_nozzle_pressure_ratio': -1,
    'Relative_nozzle_pressure_ratio': -1,
    'Static_temperature_ratio': -1,
    'Absolute_total_temperature_ratio': -1,
    'Total_total_efficiency': -1,
    'Total_static_efficiency': -1,
    'Enthalpy': 1,
    'Degree_reaction': 1,
    'Polytropic_efficiency': -1,
    'Isentropic_efficiency': -1,
    'Static_Enthalpy': -1,
    'Absolute_Enthalpy': -1,
    'Relative_Enthalpy': -1,
    'Mass_flow': -1,
    }

    return dict[parameter]

def MaxOrMIn_uq(parameter):
    dict = {
    'mean': 1,
    'var': 1,
    }

    return dict[parameter]



