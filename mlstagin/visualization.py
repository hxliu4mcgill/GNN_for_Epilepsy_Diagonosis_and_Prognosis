import os.path
import random
import numpy as np
import util
import os.path
import xlrd
import xlwt
from xlrd import open_workbook
import numpy as np
from nilearn import plotting
import matplotlib
from einops import rearrange, repeat
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.io import loadmat
import torch
import pandas as pd

def get_color(i):
    # 取对应脑区的颜色
    if 1 <= i <= 68:
        return '#696969' # dimgrey灰色         Frontal Lobe
    elif 69 <= i <= 124:
        return '#FFA07A'  # lightsalmon肉色     Temporal Lobe
    elif 125 <= i <= 162:
        return '#FFA500'  # orange              Parietal Lobe
    elif 163 <= i <= 174:
        return '#8FBC8F'  # darkseagreen        Insular Lobe
    elif 175 <= i <= 188:
        return '#FF1493'  # deeppink            Limbic Lobe
    elif 189 <= i <= 210:
        return '#40E0D0'  # 'turquoise' 蓝绿色   Occipital Lobe
    elif 211 <= i <= 214:
        return '#FF0000'  # 'red'              Amygdala杏仁核(皮下)
    elif 215 <= i <= 218:
        return '#000000'  # black              Hippocampus 海马
    elif 219 <= i <= 246:
        return '#FF0000'  #   'red'             Subcortical皮下


def analysis(argv):
    path = os.path.join(argv.targetdir, 'saliency')
    if not os.path.isdir(path):
        print('No saliency map!')
        return 0
    files = os.listdir(path)

    for i in files:
        if i.endswith('hc_x.npy'):
            hc_x = np.load(os.path.join(path, i))
            break
    for i in files:
        if i.endswith('hc_z.npy'):
            hc_z = np.load(os.path.join(path, i))
            break
    for i in files:
        if i.endswith('tle_x.npy'):
            tle_x = np.load(os.path.join(path, i))
            break
    for i in files:
        if i.endswith('tle_z.npy'):
            tle_z = np.load(os.path.join(path, i))
            break

    # plot node:

    sub_num = 29
    fold = 4
    node_num = 246
    topk = int(246 * 0.1)
    save_path = 'F:\KeYan\癫痫预后\code\FC_plot\GNN_edge_attention\{}'.format(fold)
    sub_level = True
    group_level = True
    HC_TLE_group_level = False
    input_path = r'F:\KeYan\癫痫预后\code\FC_plot\relateTmap_matrixT15_node.xls'

    # 筛选目标node
    tle_x_ = abs(tle_x.squeeze(0))
    full_sort = np.argsort(-tle_x_)
    threshold_idx = full_sort.take(np.arange(topk))
    threshold = tle_x_[threshold_idx[-1]]

    # 对应node取坐标
    axis_246 = xlrd.open_workbook(r'F:\KeYan\癫痫预后\code\FC_plot\xyz_246.xls')
    sheet = axis_246.sheet_by_index(0)
    region_axis = np.zeros((len(threshold_idx), 3))
    count = 0
    for i in threshold_idx:
        ceil_value = sheet.row_slice(i, 0, 3)
        for cols, axis in enumerate(ceil_value):
            axis = int(axis.value)
            region_axis[count][cols] = axis
        count += 1

    # fig1 = plotting.plot_markers(tle_x_, region_axis, node_threshold=threshold, node_cmap='coolwarm', alpha=1)
    # plt.show()
    # 构建边（全0）
    new_list = tle_x_[threshold_idx]
    edge_matrix = np.zeros((len(new_list), len(new_list))) + 1e-12

    # 构建节点颜色
    color_list = []
    for i in threshold_idx:
        color_list.append(get_color(i))
    edge_kwargs = {
        'linewidth': 2,  # Line width
        'color': 'none',  # Line color
        'alpha': 0.7,  # Line transparency
        'linestyle': '--',  # Line style (dashed)
    }
    fig1 = plotting.plot_connectome(edge_matrix, region_axis, node_color=color_list, edge_cmap='coolwarm',
                                   colorbar=False, edge_kwargs=edge_kwargs)
    # plt.show()


    # plot edge:
    topk = int(246 * 245 * 0.5 * 0.002)
    tle_z = abs(tle_z)
    tle_z_T = tle_z.transpose()
    atteMatrix = (tle_z_T + tle_z) / 2
    # atteMatrix = tle_z
    atteMatrix = np.triu(atteMatrix,1)
    atteMatrix = torch.tensor(atteMatrix)
    atteMatrix_ = rearrange(atteMatrix.clone(), 'n c -> (n c)')
    filter_value = 0.0
    indices_to_remove = atteMatrix_ < torch.topk(atteMatrix_, topk)[0][..., -1, None]
    atteMatrix_[indices_to_remove] = filter_value  # 对于topk之外的其他元素的logits值设为0

    # 取前topk的坐标
    val, idx = torch.topk(atteMatrix_, topk, dim=-1)
    x = torch.floor(idx / node_num)
    y = idx % node_num
    atteMatrix_ = rearrange(atteMatrix_, '(n c) -> n c', n=node_num, c=node_num)  # 矩阵化
    Tmatrix = atteMatrix_
    i = 0
    list1 = []
    while i < topk:
        list1.append(int(x[i]))
        list1.append(int(y[i]))
        i += 1
    new_list = list(set(list1))
    new_list.sort(key=list1.index)

    # 生成边矩阵
    edge_matrix = np.zeros((len(new_list), len(new_list)))
    i = 0
    val = []
    x_csv = []
    y_csv = []
    while i < topk:
        edge_matrix[new_list.index(int(x[i]))][new_list.index(int(y[i]))], \
        edge_matrix[new_list.index(int(y[i]))][new_list.index(int(x[i]))] = \
            Tmatrix[int(x[i])][int(y[i])], \
            Tmatrix[int(x[i])][int(y[i])]
        val.append(Tmatrix[int(x[i])][int(y[i])].item())
        i += 1
    count = 0
    region_axis = np.zeros((len(new_list), 3))
    # 生成颜色代表脑区
    node_list = []
    for i in new_list:
        node_list.append(get_color(i + 1))

    #点坐标
    for i in new_list:
        ceil_value = sheet.row_slice(i, 0, 3)
        for cols, axis in enumerate(ceil_value):
            cols, axis = int(cols), int(axis.value)
            region_axis[count][cols] = axis
        count += 1

    fig2 = plotting.plot_connectome(edge_matrix, region_axis, node_color=node_list, edge_cmap='coolwarm',
                                   colorbar=False)
    plt.show()
    # print()

    save_edge_excel(argv, x, y, val)

def save_edge_excel(argv, x, y, val):
    path = r'F:\KeYan\癫痫\figure\{}'.format(argv.roi)
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

    x_list_ = x.tolist()
    x_list = [int(i) for i in x_list_]
    y_list = y.tolist()
    # val_list = val.tolist()
    data = {'source_id': x_list,
            'target_id': y_list,
            'FC':val}

    df = pd.DataFrame(data)

    # 将数据帧保存到Excel文件
    df.to_csv(os.path.join(path, 'connection.csv'), index=False)

    return

if __name__=='__main__':
    # parse options and make directories
    argv = util.option.parse()
    analysis(argv)
    exit()