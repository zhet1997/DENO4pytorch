import os.path
import numpy as np
import scipy.io as sio
import paddle
import h5py
import sklearn.metrics

class DataNormer(object):
    """
        data normalization at last dimension
    """

    def __init__(self, data, method="min-max", axis=None):
        """
            data normalization at last dimension
            :param data: data to be normalized
            :param method: normalization method
            :param axis: axis to be normalized
        """
        if isinstance(data, str):
            if os.path.isfile(data):
                try:
                    self.load(data)
                except:
                    raise ValueError("the savefile format is not supported!")
            else:
                raise ValueError("the file does not exist!")
        elif type(data) is np.ndarray:
            if axis is None:
                axis = tuple(range(len(data.shape) - 1))
            self.method = method
            if method == "min-max":
                self.max = np.max(data, axis=axis)
                self.min = np.min(data, axis=axis)

            elif method == "mean-std":
                self.mean = np.mean(data, axis=axis)
                self.std = np.std(data, axis=axis)
        elif type(data) is paddle.Tensor:
            if axis is None:
                axis = tuple(range(len(data.shape) - 1))
            self.method = method
            if method == "min-max":
                self.max = np.max(data.numpy(), axis=axis)
                self.min = np.min(data.numpy(), axis=axis)

            elif method == "mean-std":
                self.mean = np.mean(data.numpy(), axis=axis)
                self.std = np.std(data.numpy(), axis=axis)
        else:
            raise NotImplementedError("the data type is not supported!")


    def norm(self, x):
        """
            input tensors
            param x: input tensors
            return x: output tensors
        """
        if paddle.is_tensor(x):
            if self.method == "min-max":
                x = 2 * (x - paddle.to_tensor(self.min, place=x.place)) \
                    / (paddle.to_tensor(self.max, place=x.place) - paddle.to_tensor(self.min, place=x.place) + 1e-10) - 1
            elif self.method == "mean-std":
                x = (x - paddle.to_tensor(self.mean, place=x.place)) / (paddle.to_tensor(self.std + 1e-10, place=x.place))
        else:
            if self.method == "min-max":
                x = 2 * (x - self.min) / (self.max - self.min + 1e-10) - 1
            elif self.method == "mean-std":
                x = (x - self.mean) / (self.std + 1e-10)

        return x

    def back(self, x):
        """
            input tensors
            param x: input tensors
            return x: output tensors
        """
        if paddle.is_tensor(x):
            if self.method == "min-max":
                x = (x + 1) / 2 * (paddle.to_tensor(self.max, place=x.device)
                                   - paddle.to_tensor(self.min, place=x.device) + 1e-10) + paddle.to_tensor(self.min,
                                                                                                     place=x.device)
            elif self.method == "mean-std":
                x = x * (paddle.to_tensor(self.std + 1e-10, place=x.device)) + paddle.to_tensor(self.mean, place=x.device)
        else:
            if self.method == "min-max":
                x = (x + 1) / 2 * (self.max - self.min + 1e-10) + self.min
            elif self.method == "mean-std":
                x = x * (self.std + 1e-10) + self.mean
        return x
    def save(self, save_path):
        """
            save the parameters to the file
            :param save_path: file path to save
        """
        import pickle
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, save_path):
        """
            load the parameters from the file
            :param save_path: file path to load
        """
        import pickle
        isExist = os.path.exists(save_path)
        if isExist:
            try:
                with open(save_path, 'rb') as f:
                    load = pickle.load(f)
                self.method = load.method
                if load.method == "mean-std":
                    self.std = load.std
                    self.mean = load.mean
                elif load.method == "min-max":
                    self.min = load.min
                    self.max = load.max
            except:
                raise ValueError("the savefile format is not supported!")
        else:
            raise ValueError("The pkl file is not exist, CHECK PLEASE!")


# reading data
class MatLoader(object):
    def __init__(self, file_path, to_paddle=True, to_cuda=False, to_float=True):
        super(MatLoader, self).__init__()

        self.to_paddle = to_paddle
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):

        try:
            self.data = sio.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_paddle:
            x = paddle.to_tensor(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_paddle(self, to_paddle):
        self.to_paddle = to_paddle

    def set_float(self, to_float):
        self.to_float = to_float


class SquareMeshGenerator(object):
    def __init__(self, real_space, mesh_size):
        """
        在多维空间中获得正交网格
        real_space shape[d, 2] 空间的上下界
        mesh_size shape[d] 每个维度的采样密度
        """
        super(SquareMeshGenerator, self).__init__()

        self.d = len(real_space)
        self.s = mesh_size[0]

        assert len(mesh_size) == self.d

        if self.d == 1:
            self.n = mesh_size[0]
            self.grid = np.linspace(real_space[0][0], real_space[0][1], int(self.n)).reshape((int(self.n), 1))
        else:
            self.n = 1
            grids = []
            for j in range(self.d):
                grids.append(np.linspace(real_space[j][0], real_space[j][1], int(mesh_size[j])))
                self.n *= mesh_size[j]

            self.grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
            # 将多个N维网格点坐标组合成一个 (N, d) 的数组

    def ball_connectivity(self, r):
        pwd = sklearn.metrics.pairwise_distances(self.grid) #计算距离矩阵
        self.edge_index = np.vstack(np.where(pwd <= r))
        self.n_edges = self.edge_index.shape[1]

        return paddle.to_tensor(self.edge_index, dtype=paddle.long)

    def gaussian_connectivity(self, sigma):
        pwd = sklearn.metrics.pairwise_distances(self.grid)
        rbf = np.exp(-pwd ** 2 / sigma ** 2)
        sample = np.random.binomial(1, rbf)
        self.edge_index = np.vstack(np.where(sample))
        self.n_edges = self.edge_index.shape[1]
        return paddle.to_tensor(self.edge_index, dtype=paddle.long)

    def get_grid(self):
        return paddle.to_tensor(self.grid, dtype=float)

    def attributes(self, f=None, theta=None):
        if f is None:
            if theta is None:
                edge_attr = self.grid[self.edge_index.T].reshape((self.n_edges, -1))
            else:
                edge_attr = np.zeros((self.n_edges, 3 * self.d))
                edge_attr[:, 0:2 * self.d] = self.grid[self.edge_index.T].reshape((self.n_edges, -1))
                edge_attr[:, 2 * self.d] = theta[self.edge_index[0]]
                edge_attr[:, 2 * self.d + 1] = theta[self.edge_index[1]]
        else:
            xy = self.grid[self.edge_index.T].reshape((self.n_edges, -1))
            if theta is None:
                edge_attr = f(xy[:, 0:self.d], xy[:, self.d:])
            else:
                edge_attr = f(xy[:, 0:self.d], xy[:, self.d:], theta[self.edge_index[0]], theta[self.edge_index[1]])

        return paddle.to_tensor(edge_attr, dtype=float)

    def get_boundary(self):
        s = self.s
        n = self.n
        boundary1 = np.array(range(0, s))
        boundary2 = np.array(range(n - s, n))
        boundary3 = np.array(range(s, n, s))
        boundary4 = np.array(range(2 * s - 1, n, s))
        self.boundary = np.concatenate([boundary1, boundary2, boundary3, boundary4])

    def boundary_connectivity2d(self, stride=1):

        boundary = self.boundary[::stride]
        boundary_size = len(boundary)
        vertice1 = np.array(range(self.n))
        vertice1 = np.repeat(vertice1, boundary_size)
        vertice2 = np.tile(boundary, self.n)
        self.edge_index_boundary = np.stack([vertice2, vertice1], axis=0)
        self.n_edges_boundary = self.edge_index_boundary.shape[1]
        return paddle.to_tensor(self.edge_index_boundary, dtype=paddle.long)

    def attributes_boundary(self, f=None, theta=None):
        # if self.edge_index_boundary == None:
        #     self.boundary_connectivity2d()
        if f is None:
            if theta is None:
                edge_attr_boundary = self.grid[self.edge_index_boundary.T].reshape((self.n_edges_boundary, -1))
            else:
                edge_attr_boundary = np.zeros((self.n_edges_boundary, 3 * self.d))
                edge_attr_boundary[:, 0:2 * self.d] = self.grid[self.edge_index_boundary.T].reshape(
                    (self.n_edges_boundary, -1))
                edge_attr_boundary[:, 2 * self.d] = theta[self.edge_index_boundary[0]]
                edge_attr_boundary[:, 2 * self.d + 1] = theta[self.edge_index_boundary[1]]
        else:
            xy = self.grid[self.edge_index_boundary.T].reshape((self.n_edges_boundary, -1))
            if theta is None:
                edge_attr_boundary = f(xy[:, 0:self.d], xy[:, self.d:])
            else:
                edge_attr_boundary = f(xy[:, 0:self.d], xy[:, self.d:], theta[self.edge_index_boundary[0]],
                                       theta[self.edge_index_boundary[1]])

        return paddle.to_tensor(edge_attr_boundary, dtype='float32')