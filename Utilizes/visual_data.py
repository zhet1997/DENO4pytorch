"""
-- coding: utf-8 --
@Time : 2022/07/24 14:20
@Author : Tianyuan Liu
@Department : Baidu ACG
@File : visual_data.py
"""
import os
import logging
import sys
import numpy as np
import matplotlib.gridspec as gridspec
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sbn
from scipy import stats
from matplotlib.animation import FuncAnimation
import matplotlib.tri as tri
import matplotlib.cm as cm
from matplotlib import ticker, rcParams
from matplotlib.ticker import MultipleLocator
import matplotlib as mpl


def adjacent_values(vals, q1, q3):
    """
    生成四分点，plot_violin
    """
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def set_axis_style(ax, labels, position=None):
    """
    生成四分点，plot_violin
    """
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    if position is None:
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xlim(0.25, len(labels) + 0.75)
    else:
        ax.set_xticks(position)
        ax.set_xlim(1.5 * position[0] - 0.5 * position[1], 1.5 * position[-1] - 0.5 * position[-2])
    ax.set_xticklabels(labels)


class TextLogger(object):
    """
    log文件记录所有打印结果
    """

    def __init__(self, filename, level=logging.INFO, stream=sys.stdout):
        self.terminal = stream
        # self.log = open(filename, 'a')

        formatter = logging.Formatter("%(levelname)s: %(asctime)s:   %(message)s",
                                      "%m-%d %H:%M:%S")
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        # logger = logging.getLogger()
        # logger.setLevel(level)
        handler = logging.FileHandler(filename)
        handler.setFormatter(formatter)
        level = logging.INFO
        level = int(level)
        handler.setLevel(level)
        logger.addHandler(handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(level)
        logger.addHandler(stream_handler)

    @staticmethod
    def info(message: str):
        # info级别的日志，绿色
        logging.info("\033[0;32m" + message + "\033[0m")

    @staticmethod
    def warning(message: str):
        # warning级别的日志，黄色
        logging.warning("\033[0;33m" + message + "\033[0m")

    @staticmethod
    def important(message: str):
        # 重要信息的日志，红色加下划线
        logging.info("\033[4;31m" + message + "\033[0m")

    @staticmethod
    def conclusion(message: str):
        # 结论级别的日志，紫红色加下划线
        logging.info("\033[4;35m" + message + "\033[0m")

    @staticmethod
    def error(message: str):
        # error级别的日志，红色
        logging.error("\033[0;31m" + "-" * 120 + '\n| ' + message + "\033[0m" + "\n" + "└" + "-" * 150)

    @staticmethod
    def debug(message: str):
        # debug级别的日志，灰色
        logging.debug("\033[0;37m" + message + "\033[0m")

    @staticmethod
    def write(message):
        """
        文本输出记录
        """
        logging.info(message)

    def flush(self):
        """
        通过
        """
        pass


class MatplotlibVision(object):
    # 主要的绘图类
    def __init__(self, log_dir, input_name=('x'), field_name=('f',)):
        """Create a summary writer logging to log_dir."""
        self.log_dir = log_dir
        # sbn.set_style('ticks')
        # sbn.set()

        self.field_name = field_name
        self.input_name = input_name

        self.font_EN = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}
        self.font_CHN = {'family': 'SimSun', 'weight': 'normal', 'size': 20}
        self.box_line_width = 1.5
        self.font_size_label = 108
        self.font_size_cb = 72
        # self._cbs = [None] * len(self.field_name) * 3
        # gs = gridspec.GridSpec(1, 1)
        # gs.update(top=0.95, bottom=0.07, left=0.1, right=0.9, wspace=0.5, hspace=0.7)
        # gs_dict = {key: value for key, value in gs.__dict__.items() if key in gs._AllowedKeys}
        # self.fig, self.axes = plt.subplots(len(self.field_name), 3, gridspec_kw=gs_dict, num=100, figsize=(30, 20))
        self.font = {'family': 'SimSun', 'weight': 'normal', 'size': 20}
        self.config = {"font.family": 'Times New Roman',
                       "font.size": 20,
                       "mathtext.fontset": 'stix',
                       "font.serif": ['SimSun'], }
        rcParams.update(self.config)

    def plot_loss(self, fig, axs, x, y, label, title=None, xylabels=('epoch', 'loss value'),color=None, linestyle='-'):
        # sbn.set_style('ticks')
        # sbn.set(color_codes=True)
        axs.semilogy(x, y, label=label,color=color,linestyle=linestyle,linewidth=2) #对数坐标
        axs.grid(True)  # 添加网格
        axs.legend(loc="best", prop=self.font)
        axs.set_xlabel(xylabels[0], fontdict=self.font)
        axs.set_ylabel(xylabels[1], fontdict=self.font)
        axs.tick_params('both', labelsize=self.font["size"], )
        axs.set_title(title, fontdict=self.font)
        # plt.pause(0.001)

    def plot_value(self, fig, axs, x, y, label, std=None, std_factor=1.0, title=None, xylabels=('x', 'y')):
        # sbn.set_style('ticks')
        # sbn.set(color_codes=True)

        axs.plot(x, y, label=label)

        if std is not None:
            std = std * std_factor
            axs.fill_between(x, y - std, y + std, alpha=0.2, label=label+'_error')
        axs.grid(True)  # 添加网格
        axs.legend(loc="best", prop=self.font)
        axs.set_xlabel(xylabels[0], fontdict=self.font)
        axs.set_ylabel(xylabels[1], fontdict=self.font)
        axs.tick_params('both', labelsize=self.font["size"], )
        axs.set_title(title, fontdict=self.font)


    def plot_curve_scatter(self, fig, axs, x, y, labelList=None,
                           title=None, xylabels=('x', 'y'),
                           colorList=None, markerList=None,
                           msList = None, mfcList = None, mecList = None,
                           xlim=None
                           ):
        # sbn.set_style('ticks')
        # sbn.set(color_codes=True)
        # check
        # if len(x.shape) == 1:
        #     x.unsqueeze(0)
        # if len(y.shape) == 1:
        #     y.unsqueeze(0)
        if colorList is not None:
            assert (len(colorList)==x.shape[0])
        if markerList is not None:
            assert (len(markerList)==x.shape[0])
        markers = ['.', 'o', '+', 'x', 's', 'd', 'v', '^', '<', '>', 'p', 'h', '*', '2', '3', '4']
        for ii, (marker, color) in enumerate(zip(markerList, colorList)):
            if marker in markers:
                if msList is not None:
                    axs.scatter(x[ii], y[ii], label=labelList[ii], color=color, marker=marker,linewidth=1,
                                s=msList[ii], facecolor=mfcList[ii], edgecolor=mecList[ii])
                else:
                    axs.scatter(x[ii], y[ii], label=labelList[ii], color=color, marker=marker, linewidth=1,
                                s=200, facecolor=color, edgecolor=None, alpha=0.5)
            else:
                axs.plot(x[ii], y[ii], label=labelList[ii], color=color, linewidth=2)
        # axs.grid(True)  # 添加网格
        axs.legend(loc="best", prop=self.font, framealpha=1)
        axs.set_xlabel(xylabels[0], fontdict=self.font)
        axs.set_ylabel(xylabels[1], fontdict=self.font)
        axs.tick_params('both', labelsize=self.font["size"], )
        # axs.set_title(title, fontdict=self.font)

        axs.set_xlim(xlim)
        axs.set_ylim([0,1])



    def plot_value_std(self, fig, axs, x, y, label, std = None, stdaxis=0, title=None, xylabels=('x', 'y'), rangeIndex=1e2):
        """
        stdaxis 表示std所在的坐标维度 x-0, y-1
        """
        num_rows, num_cols = axs.get_subplotspec().get_gridspec().get_geometry()
        print(f"Axes position: Row {num_rows}, Column {num_cols}")
        axs.plot(x, y, label=label)
        std = std * rangeIndex
        if stdaxis==0:
            plt.fill_betweenx(y, x - std / 2, x + std / 2, alpha=0.3, label='Variance Range')
        elif stdaxis==1:
            plt.fill_between(x, y - std/2, y + std/2, alpha=0.3, label='Variance Range')
        axs.grid(True)  # 添加网格
        axs.legend(loc="best", prop=self.font)
        axs.set_xlabel(xylabels[0], fontdict=self.font)
        axs.set_ylabel(xylabels[1], fontdict=self.font)
        axs.tick_params('both', labelsize=self.font["size"], )
        axs.set_title(title, fontdict=self.font)

    def plot_value_std_clean(self, fig, axs, x, y, label=None,
                             std = None, stdaxis=0, title=None,
                             xlim=[0,1], ylim=[0,1],
                             log=False,
                             xylabels=('x', 'y'), rangeIndex=1e2, color=None):
        """
        stdaxis 表示std所在的坐标维度 x-0, y-1
        """
        num_rows, num_cols = axs.get_subplotspec().get_gridspec().get_geometry()
        print(f"Axes position: Row {num_rows}, Column {num_cols}")
        if log:
            axs.semilogy(x, y, label=label, color=color, linewidth=2)
        else:
            axs.plot(x, y, label=label, color=color, linewidth=2)

        std = std * rangeIndex
        if stdaxis==0:
            axs.fill_betweenx(y, x - std / 2, x + std / 2, alpha=0.4, label='', color=color)
        elif stdaxis==1:
            axs.fill_between(x, y - std / 2, y + std/2, alpha=0.4, label='', color=color)
        axs.grid(True)  # 添加网格
        axs.set_ylim(ylim)
        # axs.set_xlim(xlim)
        axs.tick_params(axis='both', which='both', labelsize=0)
        axs.legend(loc="best", prop=self.font)
        axs.set_xlabel(xylabels[0], fontdict=self.font)
        axs.set_ylabel(xylabels[1], fontdict=self.font)
        axs.tick_params('both', labelsize=self.font["size"], )
        axs.set_title(title, fontdict=self.font)

    def plot_cumulate(self, fig, axs, data, data_neg=None, stdaxis=0, title=None):
        import matplotlib.patches as mpatches
        def plot_one_block(accumulated, d, color=None, neg=False, stdaxis=1):
            accumulated_old = accumulated.copy()
            if neg:
                accumulated -= d
            else:
                accumulated += d

            if stdaxis==1:
                p = plt.plot(np.linspace(0,1,data_length), accumulated, color=color)
                p = plt.fill_between(np.linspace(0,1,data_length), accumulated_old, accumulated, color=color, alpha=0.5)
            elif stdaxis==0:
                p = plt.plot(accumulated, np.linspace(0,1,data_length), color=color)
                p = plt.fill_betweenx(np.linspace(0,1,data_length), accumulated_old, accumulated, color=color, alpha=0.5)

            return accumulated

        # 确定数据组数和数据长度
        num_groups = data.shape[0]
        num_groups_neg = 0
        if data_neg is not None:
            num_groups_neg = data_neg.shape[0]
        data_length = data.shape[1]

        # 生成颜色列表
        index = [0,1,2,9,3,4]
        cmap = plt.cm.get_cmap('tab10')
        colors =np.take(cmap.colors,index,axis=0)
        # colors = plt.cm.get_cmap('jet').colors[:num_groups]

        # 绘制累积值曲线和填充颜色
        accumulated = np.zeros_like(data[0])
        for ii in range(num_groups):
            accumulated = plot_one_block(accumulated,
                                         data[ii], color=colors[ii], neg=False, stdaxis=stdaxis)

        if data_neg is not None:
            colors_neg = plt.cm.get_cmap('tab10').colors[num_groups:num_groups+num_groups_neg]
            accumulated = np.zeros_like(data_neg[0])
            for ii in range(num_groups_neg):
                accumulated = plot_one_block(accumulated,
                                             data_neg[ii], color=colors_neg[ii], neg=True, stdaxis=stdaxis)

        # 添加图例
        # 获取图例句柄和标签的顺序
        handles = ["0%span", "25%span", "50%span", "75%span", "100%span", "3D adjust"]
        # handles = handles[::-1]
        # plt.legend(handles)


        # 创建图例
        # legend_handles = []
        # for ii in range(num_groups):
        #     legend_handles.append(mpatches.Patch(color=colors[ii], label=handles[ii]))
        #     # legend_handles.append(plt.Line2D([], [], color=colors[ii], marker='s', linestyle='-', markersize=10, label=handles[ii]))
        #
        # # 设置图例
        # plt.legend(handles=legend_handles)
        # plt.legend(loc="upper right", prop=self.font)
        if title is not None:
            axs.set_title(title, fontdict=self.font)

        # axs.set_xlim(0,0.035)
        # axs.set_ylim(0,1)

        axs.set_xlim(0,1)
        axs.set_ylim(0,0.015)

    def plot_pie(self, fig, axs, data, title=None):

        # 确定数据组数和数据长度
        num_groups = data.shape[0]

        # 生成颜色列表
        # colors = plt.cm.get_cmap('tab20').colors[:num_groups:2]

        index = [0,1,2,9,3,4]
        cmap = plt.cm.get_cmap('tab10')
        colors =np.take(cmap.colors,index,axis=0)


        # 绘制饼图


        labels = ["0%span", "25%span", "50%span", "75%span", "100%span", "3D adjust"]
        # patches = axs.pie(data, autopct='%1.1f%%', startangle=90, colors=colors, labels=labels)
        patches = axs.pie(data, autopct='%1.1f%%', startangle=90, colors=colors)
        axs.axis=('equal')
        patches = patches[0]

        for patch in patches:
            patch.set_alpha(0.55)
        # 添加图例



        # plt.legend(labels=labels)
        # plt.legend(loc="best", prop=self.font, framealpha=1)
        # if title is not None:
        # 绘制图形    axs.set_title(title, fontdict=self.font)

        # 创建图例

    def plot_scatter(self, fig, axs, true, pred, axis=0, title=None, xylabels=('x', 'y')):
        # sbn.set(color_codes=True)

        axs.scatter(np.arange(true.shape[0]), true, marker='*')
        axs.scatter(np.arange(true.shape[0]), pred, marker='.')

        axs.grid(True)  # 添加网格
        axs.legend(loc="best", prop=self.font)
        axs.set_xlabel(xylabels[0], fontdict=self.font)
        axs.set_ylabel(xylabels[1], fontdict=self.font)
        axs.tick_params('both', labelsize=self.font["size"], )
        axs.set_title(title, fontdict=self.font)

    def plot_regression(self, fig, axs, true, pred, error_ratio=0.05,
                        title=None, xylabels=('true value', 'pred value')):
        # 所有功率预测误差与真实结果的回归直线
        # sbn.set(color_codes=True)

        max_value = max(true)  # math.ceil(max(true)/100)*100
        min_value = min(true)  # math.floor(min(true)/100)*100
        split_value = np.linspace(min_value, max_value, 11)

        split_dict = {}
        split_label = np.zeros(len(true), np.int32)
        for i in range(len(split_value)):
            split_dict[i] = str(split_value[i])
            index = true >= split_value[i]
            split_label[index] = i + 1

        axs.scatter(true, pred, marker='.', color='firebrick', linewidth=2.0)
        axs.plot([min_value, max_value], [min_value, max_value], '-', color='steelblue', linewidth=5.0)
        # 在两个曲线之间填充颜色
        axs.fill_between([min_value, max_value], [(1-error_ratio) * min_value, (1-error_ratio) * max_value],
                         [((1+error_ratio)) * min_value, ((1+error_ratio)) * max_value],
                         alpha=0.2, color='steelblue')

        # plt.ylim((min_value, max_value))
        axs.set_xlim((min_value, max_value))
        axs.set_ylim((min_value, max_value))
        axs.grid(True)  # 添加网格
        axs.set_xlabel(xylabels[0], fontdict=self.font)
        axs.set_ylabel(xylabels[1], fontdict=self.font)
        axs.tick_params('both', labelsize=self.font["size"], )
        axs.set_title(title, fontdict=self.font)
        axs.legend(['真实-预测', 'y=x', '±{:.2f}%'.format(error_ratio*100)], prop=self.font)


        # plt.ylim((-0.2, 0.2))
        # plt.pause(0.001)
    def plot_regression_dot(self, fig, axs, true, pred, title=None,label=None,
                            xylabels=('true value', 'pred value'),
                            color='r'
                            ):
        # 所有功率预测误差与真实结果的回归直线
        # sbn.set(color_codes=True)

        max_value = max(true)  # math.ceil(max(true)/100)*100
        min_value = min(true)  # math.floor(min(true)/100)*100
        split_value = np.linspace(min_value, max_value, 11)

        split_dict = {}
        split_label = np.zeros(len(true))
        for i in range(len(split_value)):
            split_dict[i] = str(split_value[i])
            index = true >= split_value[i]
            split_label[index] = i + 1

        axs.scatter(true, pred, marker='.', color=color, s=320 , linewidth=1,
                      facecolor = color, edgecolor = 'k', alpha=1, label =None
                    )

        axs.plot([min_value, max_value], [min_value, max_value], 'k--', linewidth=2.0)
        # 在两个曲线之间填充颜色
        axs.fill_between([0.995 * min_value, 1.005 * max_value], [0.995**2 * min_value, 0.995*1.005 * max_value],
                         [1.005*0.995 * min_value, 1.005**2 * max_value],
                         alpha=0.2, color='darkcyan')

        # plt.ylim((min_value, max_value))
        axs.set_xlim((0.995 * min_value, 1.005 * max_value))
        axs.set_ylim((0.995 * min_value, 1.005 * max_value))
        axs.tick_params(axis='both', which='both', labelsize=0)
        axs.grid(True)  # 添加网格
        axs.legend(loc="upper left", prop=self.font)
        axs.set_xlabel(xylabels[0], fontdict=self.font)
        axs.set_ylabel(xylabels[1], fontdict=self.font)
        axs.tick_params('both', labelsize=self.font["size"], )
        # axs.set_title(title, fontdict=self.font)

    def plot_error(self, fig, axs, error, error_ratio=0.05, title=None,
                   xylabels=('predicted relative error / %', 'distribution density')):
        # sbn.set_color_codes()
        # ax.bar(np.arange(len(error)), error*100, )

        error = pd.DataFrame(error) * 100 # 转换格式
        acc = (np.abs(np.array(error)) < error_ratio * 100).sum() / error.shape[0]
        # 绘制针对单变量的分布图
        sbn.distplot(error, bins=20, norm_hist=True, rug=True, fit=stats.norm, kde=False,
                     rug_kws={"color": "forestgreen"},
                     fit_kws={"color": "firebrick", "lw": 3},
                     hist_kws={"color": "steelblue"},
                     ax=axs)
        # plt.xlim([-1, 1])
        if title is None:
            title = '预测平均误差小于 {:.2f}% \n 占比为{:.2f}%'.format(error_ratio * 100, acc * 100)

        axs.grid(True)  # 添加网格
        # axs.legend(loc="best", prop=self.font)
        axs.set_xlabel(xylabels[0], fontdict=self.font)
        axs.set_ylabel(xylabels[1], fontdict=self.font)
        axs.tick_params('both', labelsize=self.font["size"], )
        axs.set_title(title, fontdict=self.font)

    def plot_box(self, fig, ax, data, title=None, legends=None, xlabel=None, xticks=None, bag_width=1.0):
        #绘制箱形图
        ax.set_title(title)
        ax.semilogy()
        ax.grid()
        n_vin = data.shape[-1]
        colors_map = ['#E4DACE', '#E5BB4B', '#498EAF', '#631F16']
        if len(data.shape) == 2:
            positions = np.arange(n_vin) + 1
            x_pos = None
            n_bag = 1
        else:
            n_bag = data.shape[-2]
            p = (np.linspace(0, 1, n_vin + 2) - 0.5) * bag_width
            positions = np.hstack([p[1:-1] + 0.5 + i for i in range(n_bag)]) * n_vin
            x_pos = np.arange(n_bag) * n_vin + n_vin / 2
        # parts = ax.boxplot(data.reshape(data.shape[0], -1), widths=0.5 * bag_width, positions=positions, vert=True,
        #                    patch_artist=True, )
        parts = ax.boxplot(data.reshape(data.shape[0], -1),
                           widths=0.5 * bag_width, positions=positions,
                           vert=True, patch_artist=True, )

        for i in range(n_vin):
            for j in range(n_bag):
                parts['boxes'][i + j * n_vin].set_facecolor(colors_map[i%len(colors_map)])  # violin color
                parts['boxes'][i + j * n_vin].set_edgecolor('grey')  # violin edge
                parts['boxes'][i + j * n_vin].set_alpha(0.9)
        if legends is not None:
            ax.legend(legends)
        if xticks is None:
            xticks = np.arange(n_vin * n_bag)
        ax.set_xlabel(xlabel)
        set_axis_style(ax, xticks, x_pos)

    def plot_violin(self, fig, ax, data, title=None, legends=None, xticks=None, xlabel=None, bag_width=1.0):
        ax.set_title(title)
        ax.semilogy()
        ax.grid()
        n_vin = data.shape[-1]
        colors_map = ['#E4DACE', '#E5BB4B', '#498EAF', '#631F16']
        if len(data.shape) == 2:
            positions = np.arange(n_vin) + 1
            x_pos = None
            n_bag = 1
        else:
            n_bag = data.shape[-2]
            p = (np.linspace(0, 1, n_vin + 2) - 0.5) * bag_width
            positions = np.hstack([p[1:-1] + 0.5 + i for i in range(n_bag)]) * n_vin
            x_pos = np.arange(n_bag) * n_vin + n_vin / 2

        parts = ax.violinplot(data.reshape(data.shape[0], -1), widths=0.5 * bag_width, positions=positions,
                              showmeans=False, showmedians=False, showextrema=False)

        for i in range(n_vin):
            for j in range(n_bag):
                parts['bodies'][i + j * n_vin].set_facecolor(colors_map[i%len(colors_map)])  # violin color
                parts['bodies'][i + j * n_vin].set_edgecolor('grey')  # violin edge
                parts['bodies'][i + j * n_vin].set_alpha(0.9)
        ax.legend(legends)
        quartile1, medians, quartile3 = np.percentile(data.reshape(data.shape[0], -1), [25, 50, 75], axis=0)
        whiskers = np.array([
            adjacent_values(sorted_array, q1, q3)
            for sorted_array, q1, q3 in zip(data.reshape(data.shape[0], -1), quartile1, quartile3)])
        whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

        ax.scatter(positions, medians, marker='o', color='white', s=5, zorder=3)
        ax.vlines(positions, quartile1, quartile3, color='black', linestyle='-', lw=5)
        # ax.vlines(positions, whiskers_min, whiskers_max, color='black', linestyle='-', lw=1)
        if xticks is None:
            xticks = np.arange(n_vin * n_bag)
        ax.set_xlabel(xlabel)
        set_axis_style(ax, xticks, x_pos)

    def plot_histogram(self, fig, axs, data, bins=10, range=None, color='blue', alpha=1.0):
        """
        Plot histogram on specified figure and axes.

        Parameters:
        - data: 1D array-like, input data for histogram.
        - bins: int or array_like, number of bins or bin edges.
        - range: tuple, optional, the lower and upper range of the bins.
        - color: str, optional, color of the bars.
        - alpha: float, optional, transparency of the bars.
        - fig: matplotlib.figure.Figure, optional, figure to use for plotting.
        - axs: matplotlib.axes._axes.Axes, optional, axes to use for plotting.
        """
        # Plot histogram
        axs.hist(data, bins=bins, range=range, color=color, alpha=alpha)

        # Set labels and title
        axs.set_xlabel('Value')
        axs.set_ylabel('Frequency')
        axs.set_title('Histogram')

    def plot_fields1d(self, fig, axs, real, pred, coord=None,
                      title=None, xylabels=('x coordinate', 'field'), legends=None,
                      show_channel=None):

        if len(axs.shape) == 1:
            axs = axs[None, :]

        if show_channel is None:
            show_channel = np.arange(len(self.field_name))

        if legends is None:
            legends = ['true', 'pred', 'error']

        num_channel = len(show_channel)
        name_channel = [self.field_name[i] for i in show_channel]

        for i in range(num_channel):

            fi = show_channel[i]
            ff = [real[..., fi], pred[..., fi], real[..., fi] - pred[..., fi]]
            limit = max(abs(ff[-1].min()), abs(ff[-1].max()))

            axs[i][0].cla()
            axs[i][0].plot(coord, ff[0], color='steelblue', linewidth=3, label=legends[0])
            axs[i][0].plot(coord, ff[1], '*', color='firebrick', linewidth=10, label=legends[1])
            axs[i][1].plot(coord, ff[2], color='forestgreen', linewidth=2, label=legends[2])
            for j in range(2):
                axs[i][j].legend(loc="best", prop=self.font)
                axs[i][j].set_xlabel(xylabels[0], fontdict=self.font)
                axs[i][j].set_ylabel(xylabels[1], fontdict=self.font)
                axs[i][j].tick_params('both', labelsize=self.font["size"], )
                axs[i][j].set_title(title, fontdict=self.font)

    def plot_fields_tr(self, fig, axs, real, pred, coord, edges, mask=None, cmin_max=None, fmin_max=None,
                       show_channel=None, cmaps=None, titles=None):

        if len(axs.shape) == 1:
            axs = axs[None, :]

        if show_channel is None:
            show_channel = np.arange(len(self.field_name))

        if fmin_max is None:
            fmin, fmax = real.min(axis=0), real.max(axis=0)
        else:
            fmin, fmax = fmin_max[0], fmin_max[1]

        if cmin_max is None:
            cmin, cmax = coord.min(axis=0), coord.max(axis=0)
        else:
            cmin, cmax = cmin_max[0], cmin_max[1]

        if titles is None:
            titles = ['truth', 'predicted', 'error']

        if cmaps is None:
            cmaps = ['RdYlBu_r', 'RdYlBu_r', 'coolwarm']

        x_pos = coord[:, 0]
        y_pos = coord[:, 1]

        size_channel = len(show_channel)
        name_channel = [self.field_name[i] for i in show_channel]

        for i in range(size_channel):
            fi = show_channel[i]
            ff = [real[..., fi], pred[..., fi], real[..., fi] - pred[..., fi]]
            limit = max(abs(ff[-1].min()), abs(ff[-1].max()))
            for j in range(3):
                f_true = axs[i][j].tripcolor(x_pos, y_pos, ff[j], triangles=edges, cmap=cmaps[j], shading='gouraud',
                                             antialiased=True, snap=True)

                # f_true = axs[i][j].tricontourf(triObj, ff[j], 20, cmap=cmaps[j])
                if mask is not None:
                    axs[i][j].fill(mask[:, 0], mask[:, 1], facecolor='white')
                # f_true.set_zorder(10)

                # axs[i][j].grid(zorder=0, which='both', color='grey', linewidth=1)
                axs[i][j].set_title(titles[j], fontdict=self.font_CHN)
                axs[i][j].axis([cmin[0], cmax[0], cmin[1], cmax[1]])
                # axs[i][j].tick_params(axis='x', labelsize=)
                # if i == 0:
                #     ax[i][j].set_title(titles[j], fontdict=self.font_CHN)
                cb = fig.colorbar(f_true, ax=axs[i][j])
                cb.ax.tick_params(labelsize=20)
                for l in cb.ax.yaxis.get_ticklabels():
                    l.set_family('Times New Roman')
                tick_locator = ticker.MaxNLocator(nbins=6)  # colorbar上的刻度值个数
                cb.locator = tick_locator
                cb.update_ticks()
                if j < 2:
                    f_true.set_clim(fmin[i], fmax[i])
                    cb.ax.set_title(name_channel[i], fontdict=self.font_EN, loc='center')
                else:
                    f_true.set_clim(-limit, limit)
                    cb.ax.set_title('$\mathrm{\Delta}$' + name_channel[i], fontdict=self.font_EN, loc='center')
                # 设置刻度间隔
                axs[i][j].set_aspect(1)
                # axs[i][j].xaxis.set_major_locator(MultipleLocator(0.1))
                # axs[i][j].yaxis.set_major_locator(MultipleLocator(0.1))
                # axs[i][j].xaxis.set_minor_locator(MultipleLocator(0.2))
                # axs[i][j].yaxis.set_minor_locator(MultipleLocator(0.1))
                axs[i][j].set_xlabel(r'$x$', fontdict=self.font_EN)
                axs[i][j].set_ylabel(r'$y$', fontdict=self.font_EN)
                axs[i][j].spines['bottom'].set_linewidth(self.box_line_width)  # 设置底部坐标轴的粗细
                axs[i][j].spines['left'].set_linewidth(self.box_line_width)  # 设置左边坐标轴的粗细
                axs[i][j].spines['right'].set_linewidth(self.box_line_width)  # 设置右边坐标轴的粗细
                axs[i][j].spines['top'].set_linewidth(self.box_line_width)  # 设置右边坐标轴的粗细


    def plot_fields_ms(self, fig, axs, real, pred, coord, cmin_max=None, fmin_max=None, show_channel=None,
                       cmaps=None, titles=None, limitList=None):

        if len(axs.shape) == 1:
            axs = axs[None, :]

        if show_channel is None:
            show_channel = np.arange(len(self.field_name))

        if fmin_max is None:
            fmin, fmax = real.min(axis=(0, 1)), real.max(axis=(0, 1))
        else:
            fmin, fmax = fmin_max[0], fmin_max[1]

        if coord is None:
            x = np.linspace(0, 1, real.shape[1])
            y = np.linspace(0, 1, real.shape[0])
            coord = np.stack(np.meshgrid(x, y), axis=-1)

        if cmin_max is None:
            cmin, cmax = coord.min(axis=(0, 1)), coord.max(axis=(0, 1))
        else:
            cmin, cmax = cmin_max[0], cmin_max[1]

        if titles is None:
            titles = ['truth', 'predicted', 'error']

        if cmaps is None:
            cmaps = ['RdYlBu_r', 'RdYlBu_r', 'coolwarm']

        x_pos = coord[:, :, 0]
        y_pos = coord[:, :, 1]
        size_channel = len(show_channel)
        name_channel = [self.field_name[i] for i in show_channel]

        for i in range(size_channel):

            fi = show_channel[i]
            ff = [real[..., fi], pred[..., fi], real[..., fi] - pred[..., fi]]
            if limitList is not None:
                limit = limitList[i]
            else:
                limit = max(abs(ff[-1].min()), abs(ff[-1].max()))
            #
            for j in range(3):

                axs[i][j].cla()
                f_true = axs[i][j].pcolormesh(x_pos, y_pos, ff[j], cmap=cmaps[j], shading='gouraud',
                                              antialiased=True, snap=True)
                f_true.set_zorder(10)
                axs[i][j].axis([cmin[0], cmax[0], cmin[1], cmax[1]])
                axs[i][j].axis('equal')
                # axs[i][j].grid(zorder=0, which='both', color='grey', linewidth=1)
                # axs[i][j].set_title(titles[j], fontdict=self.font_EN)
                # if i == 0:
                #     axs[i][j].set_title(titles[j], fontdict=self.font_CHN)

                cb = fig.colorbar(f_true, ax=axs[i][j], shrink=0.75)
                cb.ax.tick_params(labelsize=10)
                for l in cb.ax.yaxis.get_ticklabels():
                    l.set_family('Times New Roman')
                tick_locator = ticker.MaxNLocator(nbins=6)  # colorbar上的刻度值个数
                cb.locator = tick_locator
                cb.update_ticks()


                if j < 2:
                    f_true.set_clim(fmin[i], fmax[i])
                    cb.ax.set_title(name_channel[i], fontdict=self.font_EN, loc='center')
                else:
                    f_true.set_clim(-limit, limit)
                    cb.ax.set_title('$\mathrm{\Delta}$' + name_channel[i], fontdict=self.font_EN, loc='center')
                # 设置刻度间隔
                axs[i][j].set_xticklabels([])
                axs[i][j].set_yticklabels([])
                axs[i][j].tick_params(axis='both', which='both', length=0, labelsize=0)
                # axs[i][j].set_aspect(1)
                # axs[i][j].set_xlabel(r'$x$/m', fontdict=self.font_EN)
                # axs[i][j].set_ylabel(r'$y$/m', fontdict=self.font_EN)
                self.box_line_width = 0
                axs[i][j].spines['bottom'].set_linewidth(self.box_line_width)  # 设置底部坐标轴的粗细
                axs[i][j].spines['left'].set_linewidth(self.box_line_width)  # 设置左边坐标轴的粗细
                axs[i][j].spines['right'].set_linewidth(self.box_line_width)  # 设置右边坐标轴的粗细
                axs[i][j].spines['top'].set_linewidth(self.box_line_width)  # 设置右边坐标轴的粗细

    def plot_fields_ms_2col(self, fig, axs, real, pred, coord, cmin_max=None, fmin_max=None, show_channel=None,
                       cmaps=None, titles=None):

        if len(axs.shape) == 1:
            axs = axs[None, :]

        if show_channel is None:
            show_channel = np.arange(len(self.field_name))
        fmin = np.zeros([2,real.shape[-1]])
        fmax = np.zeros([2,real.shape[-1]])
        if fmin_max == None:
            fmin[0,:], fmax[0,:] = real.min(axis=(0, 1)), real.max(axis=(0, 1))
            fmin[1,:], fmax[1,:] = pred.min(axis=(0, 1)), pred.max(axis=(0, 1))
        else:
            fmin, fmax = fmin_max[0], fmin_max[1]

        if coord is None:
            x = np.linspace(0, 1, real.shape[1])
            y = np.linspace(0, 1, real.shape[0])
            coord = np.stack(np.meshgrid(x, y), axis=-1)

        if cmin_max == None:
            cmin, cmax = coord.min(axis=(0, 1)), coord.max(axis=(0, 1))
        else:
            cmin, cmax = cmin_max[0], cmin_max[1]

        if titles is None:
            titles = ['mean', 'std']

        if cmaps is None:
            cmaps = ['RdYlBu_r', 'jet']

        x_pos = coord[:, :, 0]
        y_pos = coord[:, :, 1]
        size_channel = len(show_channel)
        name_channel = [self.field_name[i] for i in show_channel]

        for i in range(size_channel):

            fi = show_channel[i]
            ff = [real[..., fi], pred[..., fi], real[..., fi] - pred[..., fi]]
            #
            for j in range(2):

                axs[i][j].cla()
                f_true = axs[i][j].pcolormesh(x_pos, y_pos, ff[j], cmap=cmaps[j], shading='gouraud',
                                              antialiased=True, snap=True)
                f_true.set_zorder(10)
                axs[i][j].axis([cmin[0], cmax[0], cmin[1], cmax[1]])
                # axs[i][j].axis('equal')
                # ax[i][j].grid(zorder=0, which='both', color='grey', linewidth=1)
                # axs[i][j].set_title(titles[j], fontdict=self.font_EN)
                if i == 0:
                    axs[i][j].set_title(titles[j], fontdict=self.font_CHN)

                cb = fig.colorbar(f_true, ax=axs[i][j], shrink=0.75)
                cb.ax.tick_params(labelsize=self.font['size'])
                for l in cb.ax.yaxis.get_ticklabels():
                    l.set_family('Times New Roman')
                tick_locator = ticker.MaxNLocator(nbins=6)  # colorbar上的刻度值个数
                cb.locator = tick_locator
                cb.update_ticks()

                if j < 2:
                    f_true.set_clim(fmin[j][i], fmax[j][i])
                    cb.ax.set_title(name_channel[i], fontdict=self.font_EN, loc='center')
                    # cb.ax.set_title('$\mathrm{\Delta}$' + name_channel[i], fontdict=self.font_EN, loc='center')
                # 设置刻度间隔
                axs[i][j].set_xticklabels([])
                axs[i][j].set_yticklabels([])
                axs[i][j].tick_params(axis='both', which='both', length=0, labelsize=0)
                # axs[i][j].set_aspect(1)
                # axs[i][j].set_xlabel(r'$x$/m', fontdict=self.font_EN)
                # axs[i][j].set_ylabel(r'$y$/m', fontdict=self.font_EN)
                self.box_line_width = 0
                axs[i][j].spines['bottom'].set_linewidth(self.box_line_width)  # 设置底部坐标轴的粗细
                axs[i][j].spines['left'].set_linewidth(self.box_line_width)  # 设置左边坐标轴的粗细
                axs[i][j].spines['right'].set_linewidth(self.box_line_width)  # 设置右边坐标轴的粗细
                axs[i][j].spines['top'].set_linewidth(self.box_line_width)  # 设置右边坐标轴的粗细

    def plot_fields_ms_col(self, fig, axs, real, pred, coord, cmin_max=None, fmin_max=None, show_channel=None,
                       cmaps=None, titles=None, limit=None):
        # if len(axs.shape) == 1:
        #     axs = axs[None, :]

        if show_channel is None:
            show_channel = np.arange(len(self.field_name))

        if fmin_max == None:
            fmin, fmax = real.min(axis=(0, 1)), real.max(axis=(0, 1))
        else:
            fmin, fmax = fmin_max[0], fmin_max[1]

        if cmin_max == None:
            cmin, cmax = coord.min(axis=(0, 1)), coord.max(axis=(0, 1))
        else:
            cmin, cmax = cmin_max[0], cmin_max[1]

        if titles is None:
            titles = ['truth', 'predicted', 'error']

        if cmaps is None:
            cmaps = ['RdYlBu_r', 'RdYlBu_r', 'coolwarm']

        x_pos = coord[:, :, 0]
        y_pos = coord[:, :, 1]
        size_channel = len(show_channel)
        name_channel = [self.field_name[i] for i in show_channel]

        for j in range(size_channel):

            fi = show_channel[j]
            ff = [real[..., fi], pred[..., fi], real[..., fi] - pred[..., fi]]
            if limit is None:
                limit = max(abs(ff[-1].min()), abs(ff[-1].max()))

            for i in range(3):

                axs[j][i].cla()
                f_true = axs[j][i].pcolormesh(x_pos, y_pos, ff[i], cmap=cmaps[i], shading='gouraud',
                                              antialiased=True, snap=True)
                f_true.set_zorder(10)
                axs[j][i].axis([cmin[0], cmax[0], cmin[1], cmax[1]])
                # axs[i][j].axis('equal')
                # ax[i][j].grid(zorder=0, which='both', color='grey', linewidth=1)
                axs[j][i].set_title(titles[i], fontdict=self.font_EN)
                # if i == 0:
                #     ax[i][j].set_title(titles[j], fontdict=self.font_CHN)
                cb = fig.colorbar(f_true, ax=axs[j][i], shrink=0.75) #yanse
                cb.ax.tick_params(labelsize=self.font['size'])
                for l in cb.ax.yaxis.get_ticklabels():
                    l.set_family('Times New Roman')
                tick_locator = ticker.MaxNLocator(nbins=5)  # colorbar上的刻度值个数
                cb.locator = tick_locator
                cb.update_ticks()

                axs[j][i].set_xticklabels([])
                axs[j][i].set_yticklabels([])
                if i < 2:
                    f_true.set_clim(fmin[j], fmax[j])
                    cb.ax.set_title(name_channel[j], fontdict=self.font_EN, loc='center')
                    # axs[j][i].tick_params(axis='both', which='both', length=0, width=0)

                else:
                    f_true.set_clim(-limit, limit)
                    cb.ax.set_title('$\mathrm{\Delta}$' + name_channel[j], fontdict=self.font_EN, loc='center')
                # 设置刻度间隔
                axs[j][i].set_aspect(1)
                # axs[i][j].set_xlabel(r'$x$/m', fontdict=self.font_EN)
                # axs[i][j].set_ylabel(r'$y$/m', fontdict=self.font_EN)
                self.box_line_width = 0
                axs[j][i].spines['bottom'].set_linewidth(self.box_line_width)  # 设置底部坐标轴的粗细
                axs[j][i].spines['left'].set_linewidth(self.box_line_width)  # 设置左边坐标轴的粗细
                axs[j][i].spines['right'].set_linewidth(self.box_line_width)  # 设置右边坐标轴的粗细
                axs[j][i].spines['top'].set_linewidth(self.box_line_width)  # 设置右边坐标轴的粗细

                # axs[j][i].spines['bottom'].set_visible(False)
                axs[j][i].spines['left'].set_visible(False)
                axs[j][i].spines['right'].set_visible(False)
                axs[j][i].spines['top'].set_visible(False)

    def plot_fields_grid(self, fig, axs, real, pred, fmin_max=None, show_channel=None,
                       cmaps=None, titles=None):
        if len(axs.shape) == 1:
            axs = axs[None, :]

        if show_channel is None:
            show_channel = np.arange(len(self.field_name))

        if fmin_max == None:
            fmin, fmax = real.min(axis=(0, 1)), real.max(axis=(0, 1))
        else:
            fmin, fmax = fmin_max[0], fmin_max[1]

        if titles is None:
            titles = ['truth', 'predicted', 'error']

        if cmaps is None:
            cmaps = ['RdYlBu_r', 'RdYlBu_r', 'coolwarm']

        x = np.arange(np.shape(real)[0])
        y = np.arange(np.shape(real)[1])
        x_pos, y_pos = np.meshgrid(y, x)
        size_channel = len(show_channel)
        name_channel = [self.field_name[i] for i in show_channel]

        for i in range(size_channel):

            fi = show_channel[i]
            ff = [real[..., fi], pred[..., fi], real[..., fi] - pred[..., fi]]
            limit = max(abs(ff[-1].min()), abs(ff[-1].max()))
            for j in range(3):
                axs[i][j].cla() # 清除指定的子图er
                f_true = axs[i][j].pcolormesh(x_pos, y_pos, ff[j], cmap=cmaps[j], shading='gouraud',
                                              antialiased=True, snap=True)
                f_true.set_zorder(10)
                # axs[i][j].axis('equal')
                # ax[i][j].grid(zorder=0, which='both', color='grey', linewidth=1)
                axs[i][j].set_title(titles[j], fontdict=self.font_EN)
                # if i == 0:
                #     ax[i][j].set_title(titles[j], fontdict=self.font_CHN)
                cb = fig.colorbar(f_true, ax=axs[i][j])
                cb.ax.tick_params(labelsize=self.font['size'])
                for l in cb.ax.yaxis.get_ticklabels():
                    l.set_family('Times New Roman')
                tick_locator = ticker.MaxNLocator(nbins=3)  # colorbar上的刻度值个数
                cb.locator = tick_locator
                cb.update_ticks()
                if j < 2:
                    f_true.set_clim(fmin[i], fmax[i])
                    cb.ax.set_title(name_channel[i], fontdict=self.font_EN, loc='center')
                else:
                    f_true.set_clim(-limit, limit)
                    cb.ax.set_title('$\mathrm{\Delta}$' + name_channel[i], fontdict=self.font_EN, loc='center')
                # 设置刻度间隔
                # axs[i][j].set_aspect(1)
                axs[i][j].set_xlabel(r'$x$/m', fontdict=self.font_EN)
                axs[i][j].set_ylabel(r'$y$/m', fontdict=self.font_EN)
                axs[i][j].spines['bottom'].set_linewidth(self.box_line_width)  # 设置底部坐标轴的粗细
                axs[i][j].spines['left'].set_linewidth(self.box_line_width)  # 设置左边坐标轴的粗细
                axs[i][j].spines['right'].set_linewidth(self.box_line_width)  # 设置右边坐标轴的粗细
                axs[i][j].spines['top'].set_linewidth(self.box_line_width)  # 设置右边坐标轴的粗细
    def plot_fields_am(self, fig, axs, out_true, out_pred, coord, p_id, ):
        # 输出gif动图
        fmax = out_true.max(axis=(0, 1, 2))  # 云图标尺
        fmin = out_true.min(axis=(0, 1, 2))  # 云图标尺

        def anim_update(t_id):
            # print('para:   ' + str(p_id) + ',   time:   ' + str(t_id))
            axes = self.plot_fields_ms(fig, axs, out_true[t_id], out_pred[t_id], coord, fmin_max=(fmin, fmax))
            return axes

        anim = FuncAnimation(fig, anim_update,
                             frames=np.arange(0, out_true.shape[0]).astype(np.int64), interval=200)

        anim.save(os.path.join(self.log_dir, str(p_id) + ".gif"), writer='pillow', dpi=300)


    def output_tecplot_struct(self, out_true, out_pred, coord, field_name, output_file):
        name_true = ['True_' + name for name in field_name]
        name_pred = ['Pred_' + name for name in field_name]
        name_err = ['Err_' + name for name in field_name]

        output = np.concatenate((coord, out_true, out_pred, out_true - out_pred), axis=-1)

        d1 = pd.DataFrame(output.reshape(-1, output.shape[-1]))
        f = open(output_file, "w")
        f.write("%s\n" % ('TITLE = "Element Data"'))
        if coord.shape[-1] == 1:
            f.write("%s" % ('VARIABLES = "X",'))
        elif coord.shape[-1] == 2:
            f.write("%s" % ('VARIABLES = "X","Y",'))
        else:
            f.write("%s" % ('VARIABLES = "X","Y","Z",'))

        for i in range(len(name_true)):
            f.write("%s" % ('"' + name_true[i] + '",'))

        for i in range(len(name_pred)):
            f.write("%s" % ('"' + name_pred[i] + '",'))

        for i in range(len(name_err) ):
            f.write("%s" % ('"' + name_err[i] + '",'))

        f.write("\n%s" % ('ZONE T="Turbo blade1", '))
        if len(coord.shape) == 2:
            f.write("%s" % ('I=' + str(coord.shape[0])))
        elif len(coord.shape) == 3:
            f.write("%s" % ('I=' + str(coord.shape[1]) + ', J=' + str(coord.shape[0])))
        else:
            f.write("%s" % ('I=' + str(coord.shape[0]) + ', J=' + str(coord.shape[1]) + ', K=' + str(coord.shape[2])))
        f.write("%s\n" % (', F=POINT'))
        f.close()

        d1.to_csv(output_file, index=False, mode='a', float_format="%15.5e", sep=",", header=False)


    # def output_tecplot_2d(self, out_true, out_pred, elemnets, filed_name, ):

