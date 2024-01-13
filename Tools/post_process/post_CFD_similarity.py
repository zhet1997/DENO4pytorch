
import numpy as np
import torch

class cfdPost_2d(object):
    # ===============================================================================#
    # ==========================similarity set part==================================#
    # ===============================================================================#
    def get_free_similarity(self, dof=1, scale=[-1,1], expand=1, norm=False):
        if norm:
            free_coef = np.random.normal(loc=0, scale=scale[1],size=(self.num_raw * expand, dof))
        else:
            free_coef = np.random.random([self.num_raw * expand, dof])
            free_coef = free_coef*(scale[1] - scale[0]) + scale[0]

        return free_coef

    def get_dimensional_save_dict(self):
        # M(mass) L(length) t(time) T(temprature)
        basicDimensionalDict={
            'P': np.array([1, -1, -2, 0]),
            'T': np.array([0, 0, 0, 1]),
            'V': np.array([0, 1, -1, 0]),
            'E': np.array([1, 2, -2, 0]),
            '0': np.array([0, 0, 0, 0]),
            'Rg': np.array([0, 2, -2, -1]),
            'mu': np.array([1, -1, -1, 0]),
            }
        self.basicDimensionalDict = basicDimensionalDict
        self.dimensionalSaveDict = {
                            'Static Pressure': basicDimensionalDict['P'],
                            'Relative Total Pressure': basicDimensionalDict['P'],
                            'Absolute Total Pressure': basicDimensionalDict['P'],
                            'Rotary Total Pressure': basicDimensionalDict['P'],
                            'Static Temperature': basicDimensionalDict['T'],
                            'Relative Total Temperature': basicDimensionalDict['T'],
                            'Absolute Total Temperature': basicDimensionalDict['T'],
                            'Rotary Total Temperature': basicDimensionalDict['T'],
                            'Vx': basicDimensionalDict['V'],
                            'Vy': basicDimensionalDict['V'],
                            'Vz': basicDimensionalDict['V'],
                            '|V|': basicDimensionalDict['V'],
                            '|V|^2': basicDimensionalDict['V']*2,
                            'atan(Vx/Vz)': basicDimensionalDict['0'],
                            'Flow Angle': basicDimensionalDict['0'],
                            'Rotational_speed': np.array([0, 0, -1, 0]),
                            'Density':np.array([1,-3, 0, 0]),
                            'Shroud_Gap':np.array([0, 1, 0, 0]),
                              }

    def get_dimensional_similarity(self, dof=1, scale=None, expand=1):
        free_coef = self.get_free_similarity(dof=dof, scale=scale, expand=expand)
        free_idx = slice(0, 3, 2)
        fixd_idx = slice(1, 2, 1) # the length scale is fixed in this project
        solve_idx = slice(3, 4, 1)

        solve_coef = 0 - np.sum(self.basicDimensionalDict['Rg'][free_idx] * free_coef, axis=1, keepdims=True)

        dimensional_coef = np.zeros([self.num_raw * expand, 4])
        dimensional_coef[:, free_idx] = free_coef
        # dimensional_coef[fixd_idx] = np.zeros([self.num*expand, 1])
        dimensional_coef[:, solve_idx] = solve_coef

        return dimensional_coef

    def get_dimensional_matrix(self, expand=1, scale=None,):
        if not self.sim_init:
            self.similarity_calculate_init()

        dimensional_coef = self.get_dimensional_similarity(dof=2, scale=scale, expand=expand)
        self.num = int(self.num_raw * expand)
        # for the output field
        field_dimensional_matrix = np.zeros([self.num_raw * expand, len(self.inputDict)])
        for key in self.inputDict.keys():
            dim = np.tile(self.dimensionalSaveDict[key], [self.num_raw * expand, 1]) * dimensional_coef
            dim = np.sum(dim, axis=1)
            field_dimensional_matrix[:, int(self.inputDict[key])] = dim

        boudary_dimensional_matrix = np.zeros([self.num_raw * expand, len(self.boundaryDict)])
        for key in self.boundaryDict.keys():
            dim = np.tile(self.dimensionalSaveDict[key], [self.num_raw * expand, 1]) * dimensional_coef
            dim = np.sum(dim, axis=1)
            boudary_dimensional_matrix[:, int(self.boundaryDict[key])] = dim

        return field_dimensional_matrix, boudary_dimensional_matrix

    def get_data_after_similarity(self, expand=1, scale=None, keeporder=True, log=False):
        if keeporder:
            order=1
            tile_idx = [1, expand, 1, 1, 1]
        else:
            order=0
            tile_idx = [expand, 1, 1, 1, 1]

        field_dimensional_matrix, boudary_dimensional_matrix = self.get_dimensional_matrix(expand=expand, scale=scale)
        tmp = np.tile(np.expand_dims(self.data_raw, axis=order), tile_idx).reshape(
            [-1, self.n_1d, self.n_2d, len(self.inputDict)])
        self.data_2d = self.data_similarity_operate(tmp,
                                                    np.tile(field_dimensional_matrix[:, None, None, :], [1, self.n_1d, self.n_2d, 1]),
                                                    )

        # for the input boundary condition
        tmp = np.tile(np.expand_dims(self.bouCondition_raw, axis=order), tile_idx[:2]).reshape(
            [-1, len(self.boundaryDict)])
        self.bouCondition_1d = self.data_similarity_operate(tmp, boudary_dimensional_matrix)
        # clear the old data
        if self.grid is not None:
            self.get_field_save_dict()

    @staticmethod
    def data_expand(data_raw, expand=1, keeporder=True,):
        if keeporder:
            order=1
            tile_idx = [1, expand, 1, 1, 1]
        else:
            order=0
            tile_idx = [expand, 1, 1, 1, 1]

        data_shape = data_raw.shape
        tmp = np.tile(np.expand_dims(data_raw, axis=order), tile_idx[:len(data_shape)]).reshape(
            [-1, *data_shape[1:]])

        return tmp


    #@apply_normalizer_similarity
    @staticmethod
    def data_similarity_operate(data_raw, dimensional_matrix):
        assert data_raw.shape==dimensional_matrix.shape
        dimensional_matrix = np.power(10, dimensional_matrix)
        if torch.is_tensor(data_raw):  # Check if data_raw is a PyTorch tensor
            dimensional_matrix =  torch.tensor(dimensional_matrix, dtype=data_raw.dtype, device=data_raw.device)
        data_raw = data_raw * dimensional_matrix
        return data_raw

    @staticmethod
    def data_similarity_reverse_operate(data_raw, dimensional_matrix):
        assert data_raw.shape==dimensional_matrix.shape
        dimensional_matrix = np.power(10, dimensional_matrix)
        if torch.is_tensor(data_raw):  # Check if data_raw is a PyTorch tensor
            dimensional_matrix =  torch.tensor(dimensional_matrix, dtype=data_raw.dtype, device=data_raw.device)
        data_raw = data_raw / dimensional_matrix
        return data_raw