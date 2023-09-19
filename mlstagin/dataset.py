import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from random import shuffle, randrange
import scipy.io as scio
from torch import tensor, float32, save, load
from torch.utils.data import Dataset
from nilearn.image import load_img, smooth_img, clean_img
from nilearn.maskers import NiftiLabelsMasker
from nilearn.datasets import fetch_atlas_schaefer_2018, fetch_atlas_aal, fetch_atlas_destrieux_2009, fetch_atlas_harvard_oxford
from sklearn.model_selection import StratifiedKFold

class DatasetXiangyaRest(Dataset):
    def __init__(self, sourcedir, roi, k_fold=None, laterality=False, outcome=False, smoothing_fwhm=None):
        super().__init__()
        self.filename = 'hcprest'
        self.filename += f'_roi-{roi}'
        if smoothing_fwhm is not None: self.filename += f'_fwhm-{smoothing_fwhm}'
        self.if_laterality = laterality
        self.if_outcome = outcome

        self.HC_path = 'F:\Data\Xiangya\HC'
        self.TLE_path = 'F:\Data\Xiangya\TLE'
        self.HC_FC_path = r'FC\{}\Timematrix'.format(roi)
        self.TLE_FC_path = r'FC\{}\Timematrix'.format(roi)

        HC_list = os.listdir(os.path.join(self.HC_path, self.HC_FC_path))
        self.HC_label = [0 for i in range(len(HC_list))]
        TLE_list = os.listdir(os.path.join(self.TLE_path, self.TLE_FC_path))
        self.TLE_label = [1 for i in range(len(TLE_list))]
        self.full_subject_list = HC_list + TLE_list
        self.full_label_list = self.HC_label + self.TLE_label
        self.full_label_dict = {}
        for idx in range(len(self.full_subject_list)):
            self.full_label_dict[self.full_subject_list[idx]] = self.full_label_list[idx]

        if laterality:
            self.laterality = {}
            csv_info = pd.read_csv('F:\KeYan\癫痫\\laterality.csv')
            id_ = csv_info['id'].tolist()
            laterality_ = csv_info['Laterality'].tolist()
            for idx, name in enumerate(id_):
                self.laterality[name] = laterality_[idx]
        else:
            self.laterality = None

        if outcome:
            self.outcome = {}
            csv_info = pd.read_csv('F:\KeYan\癫痫\\laterality.csv')
            id_ = csv_info['id'].tolist()
            outcome_ = csv_info['Outcome'].tolist()
            for idx, name in enumerate(id_):
                self.outcome[name] = outcome_[idx]
        else:
            self.outcome = None

        example_sub_feats_path = os.path.join(self.HC_path, self.HC_FC_path, HC_list[0])
        self.num_timepoints, self.num_nodes = scio.loadmat(example_sub_feats_path)['Mask_Timematrix'].shape

        if k_fold is None:
            self.subject_list = self.full_subject_list
        else:
            self.k_fold = StratifiedKFold(k_fold, shuffle=True, random_state=0) if k_fold is not None else None
            self.k = None

        self.num_classes = len(set(self.full_label_list))

        """数据全部加载到内存  提升训练速度"""
        self.time_series_dict = {}
        for i in range(len(self.full_subject_list)):
            subject = self.full_subject_list[i]
            label = self.full_label_dict[subject]
            if label == 0:
                label = tensor(0)
                time_series_path = os.path.join(self.HC_path, self.HC_FC_path, subject)
            elif label == 1:
                label = tensor(1)
                time_series_path = os.path.join(self.TLE_path, self.TLE_FC_path, subject)
            else:
                raise

            self.time_series_dict[subject] = scio.loadmat(time_series_path)['Mask_Timematrix']


        self.train=False

    def __len__(self):
        return len(self.subject_list) if self.k is not None else len(self.full_subject_list)


    def set_fold(self, fold, train=True):
        assert self.k_fold is not None
        self.k = fold
        self.train=train
        train_idx, test_idx = list(self.k_fold.split(self.full_subject_list, self.full_label_list))[fold]
        if train:
            shuffle(train_idx)
        self.subject_list = [self.full_subject_list[idx] for idx in train_idx] if train else [self.full_subject_list[idx] for idx in test_idx]

        # return self.subject_list, [self.full_subject_list[idx] for idx in test_idx]


    def __getitem__(self, idx):
        subject = self.subject_list[idx]
        label = self.full_label_dict[subject]
        if label==0:
            label = tensor(0)
            time_series_path = os.path.join(self.HC_path, self.HC_FC_path, subject)
        elif label==1:
            label = tensor(1)
            time_series_path = os.path.join(self.TLE_path, self.TLE_FC_path, subject)
        else:
            raise

        # timeseries = scio.loadmat(time_series_path)['Mask_Timematrix']
        """直接从内存中读取"""
        timeseries = self.time_series_dict[subject] # t * n

        timeseries = (timeseries - np.mean(timeseries, axis=0, keepdims=True)) / np.std(timeseries, axis=0, keepdims=True)

        # random flip
        if self.train:
            if(randrange(0,100) % 2 == 0):
                # 获取数组的行数和列数
                rows, cols = timeseries.shape
                # 构建一个奇偶行索引的布尔屏蔽
                mask = np.arange(cols) % 2 == 0
                # 交换奇偶行
                timeseries[:, mask], timeseries[:, ~mask] = timeseries[:, ~mask], timeseries[:, mask]

        side = -1
        if label == 1 and self.if_laterality:
            subject_ = subject.rstrip('.mat')
            subject_ = subject_.replace("_", "")
            side = self.laterality[subject_]

        outcome = -1
        if label == 1 and self.if_outcome:
            subject_ = subject.rstrip('.mat')
            subject_ = subject_.replace("_", "")
            outcome = self.outcome[subject_]

        return {'id': subject, 'idx': idx, 'timeseries': tensor(timeseries, dtype=float32), 'label': label, 'side':side, 'outcome':outcome}
        # return tensor(timeseries, dtype=float32), label

class DatasetZhengdaRest(Dataset):
    def __init__(self, sourcedir, roi, k_fold=None, target_feature='Gender', smoothing_fwhm=None):
        super().__init__()
        self.filename = 'hcprest'
        self.filename += f'_roi-{roi}'
        if smoothing_fwhm is not None: self.filename += f'_fwhm-{smoothing_fwhm}'

        self.HC_path = 'D:\data\ZD\Timeseries\HC'
        self.TLE_path = 'D:\data\ZD\Timeseries\TLE'
        self.HC_FC_path = r'FC\{}\Timematrix'.format(roi)
        self.TLE_FC_path = r'FC\{}\Timematrix'.format(roi)

        #self.HC_path = r'F:\Data\HCP\fMRI_data\Male'
        #self.TLE_path = r'F:\Data\HCP\fMRI_data\Female'
        #self.HC_FC_path = r'FC\246\Timematrix - oversamplec'
        #self.TLE_FC_path = r'FC\246\Timematrix'

        HC_list = os.listdir(os.path.join(self.HC_path, self.HC_FC_path))
        self.HC_label = [0 for i in range(len(HC_list))]
        TLE_list = os.listdir(os.path.join(self.TLE_path, self.TLE_FC_path))
        self.TLE_label = [1 for i in range(len(TLE_list))]
        self.full_subject_list = HC_list + TLE_list
        self.full_label_list = self.HC_label + self.TLE_label
        self.full_label_dict = {}
        for idx in range(len(self.full_subject_list)):
            self.full_label_dict[self.full_subject_list[idx]] = self.full_label_list[idx]

        example_sub_feats_path = os.path.join(self.HC_path, self.HC_FC_path, HC_list[0])
        self.num_timepoints, self.num_nodes = scio.loadmat(example_sub_feats_path)['Mask_Timematrix'].shape

        if k_fold is None:
            self.subject_list = self.full_subject_list
        else:
            self.k_fold = StratifiedKFold(k_fold, shuffle=True, random_state=0) if k_fold is not None else None
            self.k = None

        self.num_classes = len(set(self.full_label_list))

        """数据全部加载到内存  提升训练速度"""
        self.time_series_dict = {}
        for i in range(len(self.full_subject_list)):
            subject = self.full_subject_list[i]
            label = self.full_label_dict[subject]
            if label == 0:
                label = tensor(0)
                time_series_path = os.path.join(self.HC_path, self.HC_FC_path, subject)
            elif label == 1:
                label = tensor(1)
                time_series_path = os.path.join(self.TLE_path, self.TLE_FC_path, subject)
            else:
                raise

            self.time_series_dict[subject] = scio.loadmat(time_series_path)['Mask_Timematrix']

        self.train=False



    def __len__(self):
        return len(self.subject_list) if self.k is not None else len(self.full_subject_list)


    def set_fold(self, fold, train=True):
        assert self.k_fold is not None
        self.k = fold
        self.train=train
        train_idx, test_idx = list(self.k_fold.split(self.full_subject_list, self.full_label_list))[fold]
        if train:
            shuffle(train_idx)
        self.subject_list = [self.full_subject_list[idx] for idx in train_idx] if train else [self.full_subject_list[idx] for idx in test_idx]

        # return self.subject_list, [self.full_subject_list[idx] for idx in test_idx]


    def __getitem__(self, idx):
        subject = self.subject_list[idx]
        label = self.full_label_dict[subject]
        if label==0:
            label = tensor(0)
            time_series_path = os.path.join(self.HC_path, self.HC_FC_path, subject)
        elif label==1:
            label = tensor(1)
            time_series_path = os.path.join(self.TLE_path, self.TLE_FC_path, subject)
        else:
            raise

        # timeseries = scio.loadmat(time_series_path)['Mask_Timematrix']
        """直接从内存中读取"""
        timeseries = self.time_series_dict[subject]

        timeseries = (timeseries - np.mean(timeseries, axis=0, keepdims=True)) / np.std(timeseries, axis=0, keepdims=True)

        # random flip
        if self.train:
            if(randrange(0,100) % 2 == 0):
                # 获取数组的行数和列数
                rows, cols = timeseries.shape
                # 构建一个奇偶行索引的布尔屏蔽
                mask = np.arange(cols) % 2 == 0
                # 交换奇偶行
                timeseries[:, mask], timeseries[:, ~mask] = timeseries[:, ~mask], timeseries[:, mask]

        return {'id': subject, 'idx': idx, 'timeseries': tensor(timeseries, dtype=float32), 'label': label}
        # return tensor(timeseries, dtype=float32), label

class DatasetXYZDRest(Dataset):
    def __init__(self, sourcedir, roi, k_fold=None, target_feature='Gender', smoothing_fwhm=None):
        super().__init__()
        self.filename = 'hcprest'
        self.filename += f'_roi-{roi}'
        if smoothing_fwhm is not None: self.filename += f'_fwhm-{smoothing_fwhm}'

        """
            ZD
        """
        self.HC_path = 'D:\data\ZD\Timeseries\HC'
        self.TLE_path = 'D:\data\ZD\Timeseries\TLE'
        self.HC_FC_path = r'FC\{}\Timematrix'.format(roi)
        self.TLE_FC_path = r'FC\{}\Timematrix'.format(roi)

        HC_list = os.listdir(os.path.join(self.HC_path, self.HC_FC_path))
        self.HC_label = [0 for i in range(len(HC_list))]
        TLE_list = os.listdir(os.path.join(self.TLE_path, self.TLE_FC_path))
        self.TLE_label = [1 for i in range(len(TLE_list))]
        zd_list = HC_list + TLE_list
        zd_label = self.HC_label + self.TLE_label
        self.full_subject_list = zd_list
        self.full_label_list = zd_label
        self.full_label_dict = {}
        for idx in range(len(zd_list)):
            self.full_label_dict[zd_list[idx]] = self.full_label_list[idx]

        example_sub_feats_path = os.path.join(self.HC_path, self.HC_FC_path, HC_list[0])
        self.num_timepoints, self.num_nodes = scio.loadmat(example_sub_feats_path)['Mask_Timematrix'].shape

        """数据全部加载到内存  提升训练速度"""
        self.time_series_dict = {}
        for i in range(len(zd_list)):
            subject = zd_list[i]
            label = self.full_label_dict[subject]
            if label == 0:
                label = tensor(0)
                time_series_path = os.path.join(self.HC_path, self.HC_FC_path, subject)
            elif label == 1:
                label = tensor(1)
                time_series_path = os.path.join(self.TLE_path, self.TLE_FC_path, subject)
            else:
                raise

            self.time_series_dict[subject] = scio.loadmat(time_series_path)['Mask_Timematrix']


        """
            XY
        """
        self.HC_path = 'F:\Data\Xiangya\HC'
        self.TLE_path = 'F:\Data\Xiangya\TLE'
        self.HC_FC_path = r'FC\{}\Timematrix'.format(roi)
        self.TLE_FC_path = r'FC\{}\Timematrix'.format(roi)

        HC_list = os.listdir(os.path.join(self.HC_path, self.HC_FC_path))
        self.HC_label = [0 for i in range(len(HC_list))]
        TLE_list = os.listdir(os.path.join(self.TLE_path, self.TLE_FC_path))
        self.TLE_label = [1 for i in range(len(TLE_list))]
        xy_list = HC_list + TLE_list
        xy_label = self.HC_label + self.TLE_label
        self.full_subject_list += xy_list
        self.full_label_list += xy_label

        for idx in range(len(xy_list)):
            self.full_label_dict[xy_list[idx]] = xy_label[idx]

        if k_fold is None:
            self.subject_list = self.full_subject_list
        else:
            self.k_fold = StratifiedKFold(k_fold, shuffle=True, random_state=0) if k_fold is not None else None
            self.k = None

        self.num_classes = len(set(self.full_label_list))

        """数据全部加载到内存  提升训练速度"""

        for i in range(len(xy_list)):
            subject = xy_list[i]
            label = self.full_label_dict[subject]
            if label == 0:
                label = tensor(0)
                time_series_path = os.path.join(self.HC_path, self.HC_FC_path, subject)
            elif label == 1:
                label = tensor(1)
                time_series_path = os.path.join(self.TLE_path, self.TLE_FC_path, subject)
            else:
                raise

            timeseries = scio.loadmat(time_series_path)['Mask_Timematrix']

            # # 此处随机选取时间点将XY数据剪成ZD数据长度
            # sampling_init = randrange(timeseries.shape[0] - self.num_timepoints + 1)
            # # a = timeseries[sampling_init:sampling_init + self.num_timepoints]
            # self.time_series_dict[subject] = timeseries[sampling_init:sampling_init + self.num_timepoints]

            self.time_series_dict[subject] = timeseries

        self.train = False

    def __len__(self):
        return len(self.subject_list) if self.k is not None else len(self.full_subject_list)


    def set_fold(self, fold, train=True):
        assert self.k_fold is not None
        self.k = fold
        self.train=train
        train_idx, test_idx = list(self.k_fold.split(self.full_subject_list, self.full_label_list))[fold]
        if train:
            shuffle(train_idx)
        self.subject_list = [self.full_subject_list[idx] for idx in train_idx] if train else [self.full_subject_list[idx] for idx in test_idx]

        # return self.subject_list, [self.full_subject_list[idx] for idx in test_idx]


    def __getitem__(self, idx):
        subject = self.subject_list[idx]
        label = self.full_label_dict[subject]
        if label==0:
            label = tensor(0)
            time_series_path = os.path.join(self.HC_path, self.HC_FC_path, subject)
        elif label==1:
            label = tensor(1)
            time_series_path = os.path.join(self.TLE_path, self.TLE_FC_path, subject)
        else:
            raise

        # timeseries = scio.loadmat(time_series_path)['Mask_Timematrix']
        """直接从内存中读取"""
        timeseries = self.time_series_dict[subject]

        timeseries = (timeseries - np.mean(timeseries, axis=0, keepdims=True)) / (np.std(timeseries, axis=0, keepdims=True) + 1e-12)

        # 获取数组的行数和列数
        rows, cols = timeseries.shape# t * n
        # random flip
        if self.train:
            if(randrange(0,100) % 2 == 0):
                # 构建一个奇偶行索引的布尔屏蔽
                mask = np.arange(cols) % 2 == 0
                # 交换奇偶行
                timeseries[:, mask], timeseries[:, ~mask] = timeseries[:, ~mask], timeseries[:, mask]

        # 此处随机选取时间点将XY数据剪成ZD数据长度
        if self.train and rows > self.num_timepoints:
            sampling_init = randrange(timeseries.shape[0] - self.num_timepoints + 1)
            # a = timeseries[sampling_init:sampling_init + self.num_timepoints]
            timeseries = timeseries[sampling_init:sampling_init + self.num_timepoints]

        return {'id': subject, 'idx': idx, 'timeseries': tensor(timeseries, dtype=float32), 'label': label}
        # return tensor(timeseries, dtype=float32), label

class DatasetLOSO(Dataset):
    def __init__(self, sourcedir, roi, k_fold=None, target_feature='Gender', smoothing_fwhm=None):
        super().__init__()
        self.filename = 'hcprest'
        self.filename += f'_roi-{roi}'
        if smoothing_fwhm is not None: self.filename += f'_fwhm-{smoothing_fwhm}'

        """
            ZD
        """
        self.HC_path = 'D:\data\ZD\Timeseries\HC'
        self.TLE_path = 'D:\data\ZD\Timeseries\TLE'
        self.HC_FC_path = r'FC\{}\Timematrix'.format(roi)
        self.TLE_FC_path = r'FC\{}\Timematrix'.format(roi)

        HC_list = os.listdir(os.path.join(self.HC_path, self.HC_FC_path))
        self.HC_label = [0 for i in range(len(HC_list))]
        TLE_list = os.listdir(os.path.join(self.TLE_path, self.TLE_FC_path))
        self.TLE_label = [1 for i in range(len(TLE_list))]
        self.zd_list = HC_list + TLE_list
        zd_label = self.HC_label + self.TLE_label
        self.full_subject_list = self.zd_list[:]
        self.full_label_list = zd_label
        self.full_label_dict = {}
        for idx in range(len(self.zd_list)):
            self.full_label_dict[self.zd_list[idx]] = self.full_label_list[idx]

        example_sub_feats_path = os.path.join(self.HC_path, self.HC_FC_path, HC_list[0])
        self.num_timepoints, self.num_nodes = scio.loadmat(example_sub_feats_path)['Mask_Timematrix'].shape

        """数据全部加载到内存  提升训练速度"""
        self.time_series_dict = {}
        for i in range(len(self.zd_list)):
            subject = self.zd_list[i]
            label = self.full_label_dict[subject]
            if label == 0:
                label = tensor(0)
                time_series_path = os.path.join(self.HC_path, self.HC_FC_path, subject)
            elif label == 1:
                label = tensor(1)
                time_series_path = os.path.join(self.TLE_path, self.TLE_FC_path, subject)
            else:
                raise

            self.time_series_dict[subject] = scio.loadmat(time_series_path)['Mask_Timematrix']


        """
            XY
        """
        self.HC_path = 'F:\Data\Xiangya\HC'
        self.TLE_path = 'F:\Data\Xiangya\TLE'
        self.HC_FC_path = r'FC\{}\Timematrix'.format(roi)
        self.TLE_FC_path = r'FC\{}\Timematrix'.format(roi)

        HC_list = os.listdir(os.path.join(self.HC_path, self.HC_FC_path))
        self.HC_label = [0 for i in range(len(HC_list))]
        TLE_list = os.listdir(os.path.join(self.TLE_path, self.TLE_FC_path))
        self.TLE_label = [1 for i in range(len(TLE_list))]
        self.xy_list = HC_list + TLE_list
        xy_label = self.HC_label + self.TLE_label
        self.full_subject_list += self.xy_list
        self.full_label_list += xy_label

        for idx in range(len(self.xy_list)):
            self.full_label_dict[self.xy_list[idx]] = xy_label[idx]

        if k_fold is None:
            self.subject_list = self.full_subject_list
        else:
            self.k_fold = StratifiedKFold(k_fold, shuffle=True, random_state=0) if k_fold is not None else None
            self.k = None

        self.num_classes = len(set(self.full_label_list))

        """数据全部加载到内存  提升训练速度"""

        for i in range(len(self.xy_list)):
            subject = self.xy_list[i]
            label = self.full_label_dict[subject]
            if label == 0:
                label = tensor(0)
                time_series_path = os.path.join(self.HC_path, self.HC_FC_path, subject)
            elif label == 1:
                label = tensor(1)
                time_series_path = os.path.join(self.TLE_path, self.TLE_FC_path, subject)
            else:
                raise

            timeseries = scio.loadmat(time_series_path)['Mask_Timematrix']

            # # 此处随机选取时间点将XY数据剪成ZD数据长度
            # sampling_init = randrange(timeseries.shape[0] - self.num_timepoints + 1)
            # # a = timeseries[sampling_init:sampling_init + self.num_timepoints]
            # self.time_series_dict[subject] = timeseries[sampling_init:sampling_init + self.num_timepoints]

            self.time_series_dict[subject] = timeseries

        self.train = False

    def __len__(self):
        return len(self.subject_list) if self.k is not None else len(self.full_subject_list)

    def set_fold(self, fold, train=True):
        assert self.k_fold is not None
        self.k = fold
        self.train = train
        full_idx = list(range(len(self.full_label_list)))
        if fold == 0:
            train_idx, test_idx = full_idx[:len(self.zd_list)], full_idx[len(self.zd_list):]
        elif fold == 1:
            test_idx, train_idx = full_idx[:len(self.zd_list)], full_idx[len(self.zd_list):]
        if train:
            shuffle(train_idx)
        self.subject_list = [self.full_subject_list[idx] for idx in train_idx] if train else [self.full_subject_list[idx] for idx in test_idx]

        # return self.subject_list, [self.full_subject_list[idx] for idx in test_idx]


    def __getitem__(self, idx):
        subject = self.subject_list[idx]
        label = self.full_label_dict[subject]
        if label==0:
            label = tensor(0)
            time_series_path = os.path.join(self.HC_path, self.HC_FC_path, subject)
        elif label==1:
            label = tensor(1)
            time_series_path = os.path.join(self.TLE_path, self.TLE_FC_path, subject)
        else:
            raise

        # timeseries = scio.loadmat(time_series_path)['Mask_Timematrix']
        """直接从内存中读取"""
        timeseries = self.time_series_dict[subject]

        timeseries = (timeseries - np.mean(timeseries, axis=0, keepdims=True)) / (np.std(timeseries, axis=0, keepdims=True) + 1e-12)

        # 获取数组的行数和列数
        rows, cols = timeseries.shape# t * n
        # random flip
        if self.train:
            if(randrange(0,100) % 2 == 0):
                # 构建一个奇偶行索引的布尔屏蔽
                mask = np.arange(cols) % 2 == 0
                # 交换奇偶行
                timeseries[:, mask], timeseries[:, ~mask] = timeseries[:, ~mask], timeseries[:, mask]

        # 此处随机选取时间点将XY数据剪成ZD数据长度
        if self.train and rows > self.num_timepoints:
            sampling_init = randrange(timeseries.shape[0] - self.num_timepoints + 1)
            # a = timeseries[sampling_init:sampling_init + self.num_timepoints]
            timeseries = timeseries[sampling_init:sampling_init + self.num_timepoints]

        return {'id': subject, 'idx': idx, 'timeseries': tensor(timeseries, dtype=float32), 'label': label}
        # return tensor(timeseries, dtype=float32), label

if __name__=='__main__':
    dataset = DatasetXiangyaRest(' ', roi=246, k_fold=2)