import numpy as np
import torch.utils.data as Data
import torch.nn.functional as F
import os
import util
import random
import torch

# from models.model_wo_edge_atten import *
# from models.model_atten_wo_multihop import *
# from models.model_multihop_atten import *
from models.model_multihop_atten_topk import *
# from models.model_multihop_Markov import *

# from  model import *
from dataset import *
from tqdm import tqdm
from einops import repeat
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from dataset import *
import torch.utils.data as Data

def normalized(X):
    return (X-X.mean())/X.std()

def step(model, criterion, dyn_v, dyn_a, sampling_endpoints, t, label, reg_lambda, clip_grad=0.0, device='cpu', optimizer=None, scheduler=None, meta_train=True):
    if optimizer is None: model.eval()
    else: model.train()
    model.zero_grad()
    dyn_v = torch.autograd.Variable(dyn_v, requires_grad=True)
    dyn_v_ = dyn_v.to(device)
    dyn_a = torch.autograd.Variable(dyn_a, requires_grad=True)
    dyn_a_ = dyn_a.to(device)
    t = torch.autograd.Variable(t, requires_grad=True)
    t_ = t.to(device)
    logit, attention, latent, reg_ortho, dyn_x = model(dyn_v_, dyn_a_, t_, sampling_endpoints)
    if (label == torch.FloatTensor([0])).item():
        torch.autograd.backward(logit, torch.FloatTensor([[1., 0.]]).to(device))
    else:
        torch.autograd.backward(logit, torch.FloatTensor([[0., 1.]]).to(device))

    grad_X = dyn_v.grad.numpy()[0]
    grad_Z = dyn_a.grad.numpy()[0]
    grad_t = t.grad.numpy()[:,0,:]
    grad_X = normalized(grad_X)
    grad_Z = normalized(grad_Z)
    grad_t = normalized(grad_t)
    grad_X = np.mean(grad_X, axis=0, keepdims=True)
    grad_X = np.mean(grad_X, axis=-1)
    grad_Z = np.mean(grad_Z, axis=0)
    grad_t = np.mean(grad_t, axis=0, keepdims=True)

    return logit, 0, attention, latent, reg_ortho, grad_X, grad_Z, grad_t, label


def grad_cam(grad,feat,top_rate):
    N=len(grad)//5
    n=grad[0].shape[2]
    result=torch.zeros((116,n))
    for i in range(N):
        weight=torch.zeros(5)
        for j in range(5):
            weight[j]=(grad[i*5+j]*(grad[i*5+j]>0)).sum()
        weight=F.softmax(weight, dim=0)
        feature=torch.zeros((116,n))
        for j in range(5):
            feature+=weight[j]*(grad[i*5+j][0]>0)*grad[i*5+j][0]
        value_x,_=torch.topk(torch.abs(feature.view(-1)),int(116*n*top_rate))
        result+=(torch.abs(feature)>=value_x[-1])
    return result

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def grad(argv):
    torch.backends.cudnn.enabled = False
    node_num = 246
    # hidden_dim = argv.hidden_dim
    hidden_dim = 1
    if argv.laterality:
        Result_TLE_X_L, Result_TLE_X_R, Result_HC_X = np.zeros((hidden_dim, node_num)), np.zeros((hidden_dim, node_num)),np.zeros((hidden_dim, node_num))
        Result_TLE_Z_L, Result_TLE_Z_R, Result_HC_Z = np.zeros((node_num, node_num)),np.zeros((node_num, node_num)), np.zeros((node_num, node_num))
    else:
        Result_TLE_X, Result_HC_X = np.zeros((hidden_dim, node_num)), np.zeros((hidden_dim, node_num))
        Result_TLE_Z, Result_HC_Z = np.zeros((node_num, node_num)), np.zeros((node_num, node_num))

    if argv.outcome:
        Result_TLE_X_SF, Result_TLE_X_NSF= np.zeros((36,hidden_dim, node_num)), np.zeros((26, hidden_dim, node_num))
        Result_TLE_Z_SF, Result_TLE_Z_NSF = np.zeros((36, node_num, node_num)), np.zeros((26, node_num, node_num))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # define dataset
    if argv.dataset=='xyzd': dataset = DatasetXYZDRest(argv.sourcedir, roi=argv.roi, k_fold=argv.k_fold, smoothing_fwhm=argv.fwhm)
    elif argv.dataset=='zhengda': dataset = DatasetZhengdaRest(argv.sourcedir, roi=argv.roi, k_fold=argv.k_fold)
    elif argv.dataset == 'xiangya': dataset = DatasetXiangyaRest(argv.sourcedir, roi=argv.roi, k_fold=argv.k_fold, laterality=argv.laterality, outcome=argv.outcome)
    else: raise
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    logger = util.logger.LoggerSTAGIN(argv.k_fold, dataset.num_classes)

    sf_sub_idx = 0
    nsf_sub_idx = 0
    for k in range(argv.k_fold):
        result_hc_x = np.zeros((hidden_dim, node_num))
        result_hc_z = np.zeros((node_num, node_num))
        if argv.laterality:
            result_tle_x_l = np.zeros((hidden_dim, node_num))
            result_tle_z_l = np.zeros((node_num, node_num))
            result_tle_x_r = np.zeros((hidden_dim, node_num))
            result_tle_z_r = np.zeros((node_num, node_num))
        else:
            result_tle_x = np.zeros((hidden_dim, node_num))
            result_tle_z = np.zeros((node_num, node_num))


        os.makedirs(os.path.join(argv.targetdir, 'attention', str(k)), exist_ok=True)
        os.makedirs(os.path.join(argv.targetdir, 'saliency', str(k)), exist_ok=True)
        model = ModelSTAGIN(
            input_dim=dataset.num_nodes,
            hidden_dim=argv.hidden_dim,
            num_classes=dataset.num_classes,
            num_heads=argv.num_heads,
            num_layers=argv.num_layers,
            sparsity=argv.sparsity,
            dropout=argv.dropout,
            cls_token=argv.cls_token,
            readout=argv.readout,
            hop_num=argv.hop_num)
        model.to(device)
        model.load_state_dict(torch.load(os.path.join(argv.targetdir, 'model', str(k), 'model.pth')))
        criterion = torch.nn.CrossEntropyLoss()

        # define logging objects
        # fold_attention = {'edge_attention': [], 'time_attention': [], 'label': []}
        # summary_writer = SummaryWriter(os.path.join(argv.targetdir, 'summary', str(k), 'test'))

        logger.initialize(k)
        dataset.set_fold(k, train=False)
        # num_sample = len(dataloader.dataset)
        # if argv.outcome:
        #     result_tle_x_nsf = np.zeros((num_sample, hidden_dim, node_num))
        #     result_tle_z_nsf = np.zeros((num_sample, node_num, node_num))
        #     result_tle_x_sf = np.zeros((num_sample, hidden_dim, node_num))
        #     result_tle_z_sf = np.zeros((num_sample, node_num, node_num))

        loss_accumulate = 0.0
        reg_ortho_accumulate = 0.0
        latent_accumulate = []
        for i, x in enumerate(tqdm(dataloader, ncols=60, desc=f'k:{k}')):
            # with torch.no_grad():
            # process input data
            dyn_a, sampling_points = util.bold.process_dynamic_fc(x['timeseries'], argv.window_size, argv.window_stride)
            sampling_endpoints = [p+argv.window_size for p in sampling_points]
            if i==0: dyn_v = repeat(torch.eye(dataset.num_nodes), 'n1 n2 -> b t n1 n2', t=len(sampling_points), b=argv.minibatch_size)
            if not dyn_v.shape[1]==dyn_a.shape[1]: dyn_v = repeat(torch.eye(dataset.num_nodes), 'n1 n2 -> b t n1 n2', t=len(sampling_points), b=argv.minibatch_size)
            if len(dyn_a) < argv.minibatch_size: dyn_v = dyn_v[:len(dyn_a)]
            t = x['timeseries'].permute(1,0,2)
            label = x['label']

            logit, loss, attention, latent, reg_ortho, grad_X, grad_Z, grad_T, label = step(
                model=model,
                criterion=criterion,
                dyn_v=dyn_v,
                dyn_a=dyn_a,
                sampling_endpoints=sampling_endpoints,
                t=t,
                label=label,
                reg_lambda=argv.reg_lambda,
                clip_grad=argv.clip_grad,
                device=device,
                optimizer=None,
                scheduler=None)
            pred = logit.argmax(1)
            prob = logit.softmax(1)

            # reg_ortho_accumulate += reg_ortho.detach().cpu().numpy()
            # 每折下所有被试相加
            # 正常人
            if label == torch.FloatTensor([0]).item():
                a = grad_X + grad_T
                result_hc_x += a
                result_hc_z += grad_Z
            # 病人  而且如果分侧别
            elif argv.laterality: # L:0 R:1
                a = x['side']
                if x['side'] == 0:
                    result_tle_x_l += grad_X + grad_T
                    result_tle_z_l += grad_Z
                else:
                    result_tle_x_r += grad_X + grad_T
                    result_tle_z_r += grad_Z

            # 病人 而且 如果不分测别
            else:
                result_tle_x += grad_X + grad_T
                result_tle_z += grad_Z

            # 病人 而且 如果分术后疗效
            if label == torch.FloatTensor([1]).item() and argv.outcome: # SF:0 NSF:1
                if x['outcome'] == 0:
                    Result_TLE_X_SF[sf_sub_idx] = grad_X + grad_T
                    Result_TLE_Z_SF[sf_sub_idx] = grad_Z
                    sf_sub_idx += 1
                elif x['outcome'] == 1:
                    Result_TLE_X_NSF[nsf_sub_idx] = grad_X + grad_T
                    Result_TLE_Z_NSF[nsf_sub_idx] = grad_Z
                    nsf_sub_idx += 1
            print(sf_sub_idx)
            print(nsf_sub_idx)


        # 所有折相加
        if argv.laterality:
            Result_TLE_X_L += result_tle_x_l
            Result_TLE_X_R += result_tle_x_r
            Result_TLE_Z_L += result_tle_z_l
            Result_TLE_Z_R += result_tle_z_r
        else:
            Result_TLE_X += result_tle_x
            Result_TLE_Z += result_tle_z

        Result_HC_X += result_hc_x
        Result_HC_Z += result_hc_z

        # np.save('SA-result/ensamble/adTrained_tle_x_{}.npy'.format(k), Result_TLE_X)
        # np.save('SA-result/ensamble/adTrained_tle_z_{}.npy'.format(k), Result_TLE_Z)
        # np.save('SA-result/ensamble/adTrained_hc_x_{}.npy'.format(k), Result_HC_X)
        # np.save('SA-result/ensamble/adTrained_hc_z_{}.npy'.format(k), Result_HC_Z)
    # file_path = ".\\result01\\stagin_experiment\\stagin_experiment\\saliency"
    # # 检查文件是否存在
    # if os.path.exists(file_path):
    #     # 进行保存操作等
    #     pass
    # else:
    #     print("文件路径错误或文件不存在")
    if argv.laterality:
        np.save(os.path.join(argv.targetdir, r'saliency\adTrained_tle_x_l.npy'), Result_TLE_X_L)
        np.save(os.path.join(argv.targetdir, r'saliency\adTrained_tle_z_l.npy'), Result_TLE_Z_L)
        np.save(os.path.join(argv.targetdir, r'saliency\adTrained_tle_x_r.npy'), Result_TLE_X_R)
        np.save(os.path.join(argv.targetdir, r'saliency\adTrained_tle_z_r.npy'), Result_TLE_Z_R)
    else:
        np.save(os.path.join(argv.targetdir, r'saliency\adTrained_tle_x.npy'), Result_TLE_X)
        np.save(os.path.join(argv.targetdir, r'saliency\adTrained_tle_z.npy'), Result_TLE_Z)

    if argv.outcome:
        np.save(os.path.join(argv.targetdir, r'saliency\adTrained_tle_x_sf.npy'), Result_TLE_X_SF)
        np.save(os.path.join(argv.targetdir, r'saliency\adTrained_tle_z_sf.npy'), Result_TLE_Z_SF)
        np.save(os.path.join(argv.targetdir, r'saliency\adTrained_tle_x_nsf.npy'), Result_TLE_X_NSF)
        np.save(os.path.join(argv.targetdir, r'saliency\adTrained_tle_z_nsf.npy'), Result_TLE_Z_NSF)

    np.save(os.path.join(argv.targetdir, r'saliency\adTrained_hc_x.npy'), Result_HC_X)
    np.save(os.path.join(argv.targetdir, r'saliency\adTrained_hc_z.npy'), Result_HC_Z)
    return


# device=torch.device('cpu')
# seed = 0
# setup_seed(seed)
# cpac_root='/media/dm/0001A094000BF891/Yazid/ABIDEI_CPAC/'
# smri_root='/media/dm/0001A094000BF891/Yazid/ABIDEI_sMRI/'
# nan_subid=np.load('nan_subid.npy').tolist()
# aal=np.load('Atlas/AAL.npy').tolist()
# Lobe_aal=np.load('SA-result/Lobe_aal.npy',allow_pickle=True).tolist()
# Lobe={}
# Lobe['Central']=[1,2,57,58,17,18]
# Lobe['Frontal']=[3,4,5,6,7,8,9,10,11,12,13,14,15,16,19,20,21,22,23,24,25,26,27,28,69,70]
# Lobe['Temporal']=[79,80,81,82,85,86,89,90]
# Lobe['Parietal']=[59,60,61,62,63,64,65,66,67,68]
# Lobe['Occipital']=[43,44,45,46,47,48,49,50,51,52,53,54,55,56]
# Lobe['Limbic']=[31,32,33,34,35,36,37,38,39,40,83,84,87,88]
# Lobe['Insula']=[29,30]
# Lobe['Subcortical']=[41,42,71,72,73,74,75,76,77,78]
# Lobe['Cerebelum']=[91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108]
# Lobe['Vermis']=[109,110,111,112,113,114,115,116]

def saliency_map(argv):
    # parse options and make directories
    # argv = util.option.parse()


    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
    setup_seed(argv.seed)
    grad(argv)
    # exit(0)


    # ## gradient-basedfor
    # featuresMap=list()
    # # all_site=os.listdir(cpac_root)
    # Result_ASD_X,Result_TDC_X=np.zeros((116,3)),np.zeros((116,3))
    # Result_TDC_Z,Result_ASD_Z=np.zeros((116,116)),np.zeros((116,116))
    # vote = 13
    # for i in range(vote):
    #     for index in range(10):
    #         PATH='/media/dm/0001A094000BF891/Yazid/SAVEDModels/ensamble/models_{}_{}'.format(i,index)
    #         model=NEResGCN(5)
    #         model.load_state_dict(torch.load(PATH))
    #         train_asd,train_tdc=data_arange(all_site,fmri_root=cpac_root,smri_root=smri_root,nan_subid=nan_subid)
    #         trainset=dataset(site=all_site,fmri_root=cpac_root,smri_root=smri_root,ASD=train_asd,TDC=train_tdc)
    #         trainloader=DataLoader(trainset,batch_size=1,shuffle=True,drop_last=True)
    #         result_asd_x,result_asd_z,result_tdc_x,result_tdc_z=gradient(device,model,trainloader)
    #         Result_ASD_X+=result_asd_x
    #         Result_ASD_Z+=result_asd_z
    #         Result_TDC_X+=result_tdc_x
    #         Result_TDC_Z+=result_tdc_z
    # np.save('SA-result/ensamble/adTrained_asd_x_{}.npy'.format(vote),Result_ASD_X)
    # np.save('SA-result/ensamble/adTrained_asd_z_{}.npy'.format(vote),Result_ASD_Z)
    # np.save('SA-result/ensamble/adTrained_tdc_x_{}.npy'.format(vote),Result_TDC_X)
    # np.save('SA-result/ensamble/adTrained_tdc_z_{}.npy'.format(vote),Result_TDC_Z)


