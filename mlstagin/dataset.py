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
    def __init__(self, sourcedir, roi, k_fold=None, target_feature='Gender', smoothing_fwhm=None):
        super().__init__()
        self.filename = 'hcprest'
        self.filename += f'_roi-{roi}'
        if smoothing_fwhm is not None: self.filename += f'_fwhm-{smoothing_fwhm}'

        # self.HC_path = r'F:\Data\COBRE\246\hc'
        # self.TLE_path = r'F:\Data\COBRE\246\sz'
        # self.HC_FC_path = r'FC\264\Timematrix'
        # self.TLE_FC_path = r'FC\264\Timematrix'

        self.HC_path = 'F:\Data\Xiangya\HC'
        self.TLE_path = 'F:\Data\Xiangya\TLE'
        self.HC_FC_path = r'FC\{}\Timematrix'.format(roi)
        self.TLE_FC_path = r'FC\{}\Timematrix'.format(roi)

        #self.HC_path = 'D:\data\ZD\Timeseries\HC'
        #self.TLE_path = 'D:\data\ZD\Timeseries\TLE'
        #self.HC_FC_path = r'FC\246\Timematrix'
        #self.TLE_FC_path = r'FC\246\Timematrix'

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
                mask = np.arange(rows) % 2 == 0
                # 交换奇偶行
                timeseries[mask], timeseries[~mask] = timeseries[~mask], timeseries[mask]

        return {'id': subject, 'idx': idx, 'timeseries': tensor(timeseries, dtype=float32), 'label': label}
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
                mask = np.arange(rows) % 2 == 0
                # 交换奇偶行
                timeseries[mask], timeseries[~mask] = timeseries[~mask], timeseries[mask]

        return {'id': subject, 'idx': idx, 'timeseries': tensor(timeseries, dtype=float32), 'label': label}
        # return tensor(timeseries, dtype=float32), label


class DatasetXYZDRest(Dataset):
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
                mask = np.arange(rows) % 2 == 0
                # 交换奇偶行
                timeseries[mask], timeseries[~mask] = timeseries[~mask], timeseries[mask]

        return {'id': subject, 'idx': idx, 'timeseries': tensor(timeseries, dtype=float32), 'label': label}
        # return tensor(timeseries, dtype=float32), label


class DatasetHCPRest(Dataset):
    def __init__(self, sourcedir, roi, k_fold=None, target_feature='Gender', smoothing_fwhm=None):
        super().__init__()
        self.filename = 'hcprest'
        self.filename += f'_roi-{roi}'
        if smoothing_fwhm is not None: self.filename += f'_fwhm-{smoothing_fwhm}'

        # if roi=='schaefer': self.roi = fetch_atlas_schaefer_2018(data_dir=os.path.join(sourcedir, 'roi'))
        # elif roi=='aal': self.roi = fetch_atlas_aal(data_dir=os.path.join(sourcedir, 'roi'))
        # elif roi=='destrieux': self.roi = fetch_atlas_destrieux_2009(data_dir=os.path.join(sourcedir, 'roi'))
        # elif roi=='harvard_oxford': self.roi = fetch_atlas_harvard_oxford(atlas_name='cort-maxprob-thr25-2mm', data_dir=os.path.join(sourcedir, 'roi'))

        if os.path.isfile(os.path.join(sourcedir, f'{self.filename}.pth')):
            self.timeseries_dict = load(os.path.join(sourcedir, f'{self.filename}.pth'))
        else:
            roi_masker = NiftiLabelsMasker(load_img(self.roi['maps']))
            self.timeseries_dict = {}
            img_list = [f for f in os.listdir(os.path.join(sourcedir, 'img', 'REST')) if f.endswith('nii.gz')]
            img_list.sort()
            for img in tqdm(img_list, ncols=60): # tqdm可以产生进度条
                id = img.split('.')[0]
                timeseries = roi_masker.fit_transform(load_img(os.path.join(sourcedir, 'img', 'REST', img)))
                # if not len(timeseries) == 1200: continue
                if not len(timeseries) == 1190: continue
                self.timeseries_dict[id] = timeseries
            save(self.timeseries_dict, os.path.join(sourcedir, f'{self.filename}.pth'))

        self.num_timepoints, self.num_nodes = list(self.timeseries_dict.values())[0].shape
        self.full_subject_list = list(self.timeseries_dict.keys())
        if k_fold is None:
            self.subject_list = self.full_subject_list
        else:
            self.k_fold = StratifiedKFold(k_fold, shuffle=True, random_state=0) if k_fold is not None else None
            self.k = None

        behavioral_df = pd.read_csv(os.path.join(sourcedir, 'behavioral', 'hcp.csv')).set_index('Subject')[target_feature]
        self.num_classes = len(behavioral_df.unique())
        self.behavioral_dict = behavioral_df.to_dict()
        self.full_label_list = [self.behavioral_dict[int(subject)] for subject in self.full_subject_list]


    def __len__(self):
        return len(self.subject_list) if self.k is not None else len(self.full_subject_list)


    def set_fold(self, fold, train=True):
        assert self.k_fold is not None
        self.k = fold
        train_idx, test_idx = list(self.k_fold.split(self.full_subject_list, self.full_label_list))[fold]
        if train: shuffle(train_idx)
        self.subject_list = [self.full_subject_list[idx] for idx in train_idx] if train else [self.full_subject_list[idx] for idx in test_idx]


    def __getitem__(self, idx):
        subject = self.subject_list[idx]
        timeseries = self.timeseries_dict[subject]
        timeseries = (timeseries - np.mean(timeseries, axis=0, keepdims=True)) / np.std(timeseries, axis=0, keepdims=True)
        label = self.behavioral_dict[int(subject)]

        if label=='F':
            label = tensor(0)
        elif label=='M':
            label = tensor(1)
        else:
            raise

        return {'id': subject, 'timeseries': tensor(timeseries, dtype=float32), 'label': label}


class DatasetHCPTask(Dataset):
    def __init__(self, sourcedir, roi, dynamic_length=None, k_fold=None, smoothing_fwhm=None):
        super().__init__()
        self.filename = 'hcptest'
        self.filename += f'_roi-{roi}'
        if smoothing_fwhm is not None: self.filename += f'_fwhm-{smoothing_fwhm}'

        if roi=='schaefer': self.roi = fetch_atlas_schaefer_2018(data_dir=os.path.join(sourcedir, 'roi'))
        elif roi=='aal': self.roi = fetch_atlas_aal(data_dir=os.path.join(sourcedir, 'roi'))
        elif roi=='destrieux': self.roi = fetch_atlas_destrieux_2009(data_dir=os.path.join(sourcedir, 'roi'))
        elif roi=='harvard_oxford': self.roi = fetch_atlas_harvard_oxford(atlas_name='cort-maxprob-thr25-2mm', data_dir=os.path.join(sourcedir, 'roi'))

        task_timepoints = {'EMOTION': 176, 'GAMBLING': 253, 'LANGUAGE': 316, 'MOTOR': 284, 'RELATIONAL': 232, 'SOCIAL': 274, 'WM': 405}
        self.sourcedir = sourcedir
        self.dynamic_length = dynamic_length
        self.task_list = list(task_timepoints.keys())
        self.task_list.sort()
        print(self.task_list)

        if os.path.isfile(os.path.join(sourcedir, f'hcptask_roi-{roi}.pth')):
            self.timeseries_list, self.label_list = load(os.path.join(sourcedir, f'hcptask_roi-{roi}.pth'))
        else:
            roi_masker = NiftiLabelsMasker(load_img(self.roi['maps']))
            self.timeseries_list = []
            self.label_list = []
            for task in self.task_list:
                img_list = [f for f in os.listdir(os.path.join(sourcedir, 'img', 'TASK', task)) if f.endswith('nii.gz')]
                img_list.sort()
                for subject in tqdm(img_list, ncols=60, desc=f'prep:{task.lower()[:3]}'):
                    timeseries = roi_masker.fit_transform(load_img(os.path.join(self.sourcedir, 'img', 'TASK', task, subject)))
                    if not len(timeseries)==task_timepoints[task]:
                        print(f"short timeseries: {len(timeseries)}")
                        continue
                    self.timeseries_list.append(timeseries)
                    self.label_list.append(task)
            save((self.timeseries_list, self.label_list), os.path.join(sourcedir, f'hcptask_roi-{roi}.pth'))

        if k_fold:
            self.k_fold = StratifiedKFold(k_fold, shuffle=True, random_state=0)
            self.k = None

        self.num_nodes = self.timeseries_list[0].shape[1]
        self.num_classes = len(set(self.label_list))
        self.train = None

    def __len__(self):
        return len(self.fold_idx) if self.k is not None else len(self.timeseries_list)


    def set_fold(self, fold, train=True):
        assert self.k_fold is not None
        self.k = fold
        train_idx, test_idx = list(self.k_fold.split(self.timeseries_list, self.label_list))[fold]
        if train:
            shuffle(train_idx)
            self.fold_idx = train_idx
            self.train = True
        else:
            self.fold_idx = test_idx
            self.train = False

    def __getitem__(self, idx):
        timeseries = self.timeseries_list[self.fold_idx[idx]]
        timeseries = (timeseries - np.mean(timeseries, axis=0, keepdims=True)) / np.std(timeseries, axis=0, keepdims=True)
        if not self.dynamic_length is None:
            if self.train:
                sampling_init = randrange(len(timeseries)-self.dynamic_length)
                timeseries = timeseries[sampling_init:sampling_init+self.dynamic_length]
        task = self.label_list[self.fold_idx[idx]]

        for task_idx, _task in enumerate(self.task_list):
            if task == _task:
                label = task_idx

        return {'timeseries': tensor(timeseries, dtype=float32), 'label': tensor(label)}
