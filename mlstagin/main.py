import random
import numpy as np
import util
# from experiment import train, test
# from experiment_adversarial_pgd import train, test
# from experiment_adversarial_fgsm import train, test
from experiment_adversarial_fgsm_con import train, test
# from experiment_meta2 import train, test
# from experiment_meta import train, test
from saliency_map import saliency_map
from visualization import analysis
from analysis import analyze
import torch

if __name__=='__main__':
    # parse options and make directories
    argv = util.option.parse()

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
    setup_seed(argv.seed)
    # step1
    # if not argv.no_train: train(argv)
    # step2
    # if not argv.no_test: test(argv)
    # step3
    saliency_map(argv)
    # step4
    # analysis(argv, endwith='')

    # if not argv.no_analysis and argv.roi=='schaefer': analyze(argv)

    exit(0)

