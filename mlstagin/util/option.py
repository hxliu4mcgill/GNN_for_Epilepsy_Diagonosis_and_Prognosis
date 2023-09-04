import os
import csv
import argparse

def parse():
    parser = argparse.ArgumentParser(description='SPATIO-TEMPORAL-ATTENTION-GRAPH-ISOMORPHISM-NETWORK')

    parser.add_argument('-s', '--seed', type=int, default=10)
    # parser.add_argument('-n', '--exp_name', type=str, default='zd_stagin_experiment_seed10_win36s_stride4')
    parser.add_argument('-n', '--exp_name', type=str, default='zd_stagin_experiment_wiz_con_1e-4')

    parser.add_argument('--dataset', type=str, default='zhengda', choices=['xiangya', 'zhengda', 'xyzd'])
    # parser.add_argument('--roi', type=str, default='schaefer', choices=['scahefer', 'aal', 'destrieux', 'harvard_oxford'])
    parser.add_argument('--roi', type=int, default=246, choices=[116, 246])
    parser.add_argument('--fwhm', type=float, default=None)

    parser.add_argument('--pretrain', type=bool, default=False)

    # XY
    # parser.add_argument('--window_size', type=int, default=50)
    # parser.add_argument('--window_stride', type=int, default=3)
    # parser.add_argument('--dynamic_length', type=int, default=300)

    # ZD
    parser.add_argument('--window_size', type=int, default=36)
    parser.add_argument('--window_stride', type=int, default=2)
    parser.add_argument('--dynamic_length', type=int, default=200)

    parser.add_argument('-k', '--k_fold', type=int, default=5)
    parser.add_argument('-b', '--minibatch_size', type=int, default=8)
    parser.add_argument('-k_s', '--k_shots', type=int, default=6)

    parser.add_argument('-ds', '--sourcedir', type=str, default='./data')
    parser.add_argument('-dt', '--targetdir', type=str, default='./result')


    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--inner_lr', type=float, default=0.001)
    parser.add_argument('--max_lr', type=float, default=0.001)
    parser.add_argument('--reg_lambda', type=float, default=0.00001)
    # parser.add_argument('--reg_lambda', type=float, default=0)
    parser.add_argument('--clip_grad', type=float, default=0.0)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--sparsity', type=int, default=30)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--readout', type=str, default='mean', choices=['garo', 'sero', 'sum', 'mean'])
    parser.add_argument('--cls_token', type=str, default='mean', choices=['sum', 'mean', 'param'])
    parser.add_argument('--hop_num', type=int, default=4)

    parser.add_argument('--num_clusters', type=int, default=7)
    parser.add_argument('--subsample', type=int, default=50)

    parser.add_argument('--no_train', action='store_true')
    parser.add_argument('--no_test', action='store_true')
    parser.add_argument('--no_analysis', action='store_true')
    parser.add_argument('--val', type=bool, default=False)

    argv = parser.parse_args()
    argv.targetdir = os.path.join(argv.targetdir, argv.exp_name)
    os.makedirs(argv.targetdir, exist_ok=True)
    with open(os.path.join(argv.targetdir, 'argv.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(vars(argv).items())
    return argv


def parse():
    parser = argparse.ArgumentParser(description='SPATIO-TEMPORAL-ATTENTION-GRAPH-ISOMORPHISM-NETWORK')
    parser.add_argument('-s', '--seed', type=int, default=0)
    # parser.add_argument('-n', '--exp_name', type=str, default='zd_stagin_experiment_seed10_win36s_stride4')
    parser.add_argument('-n', '--exp_name', type=str, default='xy_stagin_experiment_wiz_con_1e-2')
    parser.add_argument('-k', '--k_fold', type=int, default=5)
    parser.add_argument('-b', '--minibatch_size', type=int, default=8)
    parser.add_argument('-k_s', '--k_shots', type=int, default=6)

    parser.add_argument('-ds', '--sourcedir', type=str, default='./data')
    parser.add_argument('-dt', '--targetdir', type=str, default='./result')

    parser.add_argument('--dataset', type=str, default='xiangya', choices=['xiangya', 'zhengda', 'xyzd'])
    # parser.add_argument('--roi', type=str, default='schaefer', choices=['scahefer', 'aal', 'destrieux', 'harvard_oxford'])
    parser.add_argument('--roi', type=int, default=246, choices=[116, 246])
    parser.add_argument('--fwhm', type=float, default=None)

    parser.add_argument('--pretrain', type=bool, default=False)

    # XY
    parser.add_argument('--window_size', type=int, default=50)
    parser.add_argument('--window_stride', type=int, default=3)
    parser.add_argument('--dynamic_length', type=int, default=300)

    # ZD
    # parser.add_argument('--window_size', type=int, default=36)
    # parser.add_argument('--window_stride', type=int, default=4)
    # parser.add_argument('--dynamic_length', type=int, default=200)

    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--inner_lr', type=float, default=0.001)
    parser.add_argument('--max_lr', type=float, default=0.001)
    parser.add_argument('--reg_lambda', type=float, default=0.00001)
    # parser.add_argument('--reg_lambda', type=float, default=0)
    parser.add_argument('--clip_grad', type=float, default=0.0)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--sparsity', type=int, default=30)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--readout', type=str, default='mean', choices=['garo', 'sero', 'sum', 'mean'])
    parser.add_argument('--cls_token', type=str, default='mean', choices=['sum', 'mean', 'param'])
    parser.add_argument('--hop_num', type=int, default=4)

    parser.add_argument('--num_clusters', type=int, default=7)
    parser.add_argument('--subsample', type=int, default=50)

    parser.add_argument('--no_train', action='store_true')
    parser.add_argument('--no_test', action='store_true')
    parser.add_argument('--no_analysis', action='store_true')
    parser.add_argument('--val', type=bool, default=False)

    argv = parser.parse_args()
    argv.targetdir = os.path.join(argv.targetdir, argv.exp_name)
    os.makedirs(argv.targetdir, exist_ok=True)
    with open(os.path.join(argv.targetdir, 'argv.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(vars(argv).items())
    return argv
