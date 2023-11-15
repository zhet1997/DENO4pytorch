import os
import numpy as np
import pyDOE
from scipy.stats import norm, moment, skew, kurtosis
from scipy import mean, var
import matplotlib.pyplot as plt


class Turbo_UQLab(object):
    def __init__(self, problem, evaluate, work_path=None):
        self.problem = problem
        self.evaluate = evaluate # a mapping from input to output

        if work_path is None:
            work_path = os.path.join('..', 'work')
        self.work_path = work_path

        self.check_problem()
        self.check_evaluate()
        assert self.num_var < self.num_input
        for name in self.names_var:
            assert name in self.names_input, print(name)

        self.get_slice()


    def check_problem(self):
        self.num_var = self.problem['num_vars']
        if 'names' not in self.problem:
            self.problem.update({'names': [f'x{i}' for i in range(self.num_var)]})
        self.names_var = self.problem['names']
        if 'bounds' not in self.problem:
            self.problem.update({'bounds': [[0, 1]] * self.num_var})
        self.bounds_var = np.array(self.problem['bounds'])

    def check_evaluate(self):
        self.input_shape = self.evaluate['input_shape']
        self.num_input = np.prod(self.input_shape)
        if 'names' not in self.evaluate:
            self.evaluate.update({'names': [f'x{i}' for i in range(self.num_input)]})
        self.names_input = self.evaluate['names']
        self.output_shape = self.evaluate['output_shape']
        self.evaluate_func = self.evaluate['evaluate_func']
        if 'reference' not in self.evaluate:
            self.evaluate.update({'reference': [0.5]*self.num_input})
        self.reference = self.evaluate['reference']

    #===============================================================================#
    #=======================sample distrubute generation============================#
    #===============================================================================#
    def sample_generate(self, number,
                        dist='uniform', #'norm'
                        generate='random', #'lhs', 'mesh'
                        paradict=dict(),
                        ):
        if dist == 'norm':
            if 'mu' not in paradict:
                paradict.update({'mu': 0})
            if 'sigma' not in paradict:
                paradict.update({'sigma': 1})

        if generate=='lhs':
            samples = pyDOE.lhs(self.num_var, samples=number, criterion='maximin')
            if dist=='norm':
                samples = norm.ppf(samples, loc=paradict['mu'], scale=paradict['sigma'])
        elif generate=='mesh':
            points = paradict['points']
            real_number = np.prod(np.array(points))
            assert real_number < 1e8, print('the node number of the generating mesh is too big')
            if self.num_var == 1:
                samples = np.linspace(0, 1, int(points[0])).reshape(-1, 1)
            else:
                grids = []
                for j in range(self.num_var):
                    grids.append(np.linspace(0, 1, int(points[j])))
                samples = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
        elif generate=='random':
            # The distribution of different dimensions is independent
            samples_1d_list = []
            for ii in range(self.num_var):
                if dist == 'uniform':
                    samples_1d_list.append(np.random.uniform(low=0, high=1, size=number).reshape(-1,1))
                elif dist=='norm':
                    samples_1d_list.append(np.random.normal(loc=paradict['mu'], scale=paradict['sigma'], size=number).reshape(-1,1))
                else:
                    assert False
            samples = np.concatenate(samples_1d_list, axis=1)
        else:
            assert False

        return samples

    # ===============================================================================#
    # =========================sample values evaluration=============================#
    # ===============================================================================#
    def get_slice(self):
        slice_index = []
        for ii in range(self.num_var):
            slice_index.append(self.names_input.index(self.names_var[ii]))
        self.slice_index = slice_index

    def input_transformer(self, x_var):
        x_input = np.tile(self.reference, [x_var.shape[0],1])
        x_input[:, self.slice_index] = x_var

        return x_input


    def value_evaluate(self, x_var):
        return self.evaluate_func(self.input_transformer(x_var))

    # ===============================================================================#
    # =========================moment calculate and cvpe=============================#
    # ===============================================================================#
    def moment_calculate(self, data,
                         type='mean',# 'mean':1, 'var':2, 'skew':3, 'kurt':4
                         channel=False,
                         ):
        if type=='mean' or type==1:
            rst =  mean(data, axis=0, keepdims=True)
        elif type == 'var' or type == 2:
            rst = var(data, axis=0, keepdims=True)
        elif type == 'skew' or type == 3:
            rst = skew(data, axis=0, keepdims=True)
        elif type == 'kurt' or type == 4:
            rst = kurtosis(data, axis=0, keepdims=True)
        else:
            assert False
        return rst
    def moment_convergence(self,
                           number_ini=1000,
                           number_iter=100,
                           max_iter=20,
                           type='mean',
                           rel_error=0.001,
                           abs_error=0.001):
        samples = self.sample_generate(number_ini, dist='uniform', generate='lhs')
        result = self.value_evaluate(samples)
        target = self.moment_calculate(result)

        for _ in range(max_iter):
            samples_add = self.sample_generate(number_iter, dist='uniform', generate='random')
            result_add = self.value_evaluate(samples_add)
            result = np.concatenate((result, result_add), axis=0)
            target_new = self.moment_calculate(result)

            if np.mean(self.rel_error(target, target_new), axis=target.shape)>rel_error:
                continue
            else:
                return target_new

        return target_new

    @staticmethod
    def rel_error(x, y):
        return np.abs(x-y)/y

    @staticmethod
    def abs_error(x, y):
        return np.abs(x-y)
    # ===============================================================================#
    # ============================result data output=================================#
    # ===============================================================================#


if __name__ == "__main__":
    var_dim = 10
    eval_dim = 100
    problem = {
        'num_vars': var_dim,  # 参数数量
        'names': [f'x{i*3}' for i in range(var_dim)],  # 参数名称
        'bounds': [[0, 1]] * var_dim,  # 参数范围
    }

    evaulate = {
        'input_shape': [100,],
        'names' : [f'x{i}' for i in range(eval_dim)],
        'output_shape': [128, 128, 1],
        'evaluate_func': lambda x: np.sum(np.power(x, 1), axis=1, keepdims=True),
        # 'evaluate_func': lambda x: np.tile(np.sum(np.power(x,1), axis=1, keepdims=True)[:,np.newaxis,:], [1,3,2]),
        'reference': [0.5] * eval_dim,
    }

    para = {
        'mu': 5,
        'sigma': 3,
    }

    SA = Turbo_UQLab(problem, evaulate)
    data = SA.sample_generate(1000, dist='norm',generate='lhs',paradict=para)
    print(data)
    rst = SA.value_evaluate(data)
    print(rst)
    sa = SA.moment_calculate(rst, type=1)
    print(sa)
    plt.hist(rst, bins=30, density=True, alpha=0.6, color='b', label='Monte Carlo Samples')
    plt.show()

