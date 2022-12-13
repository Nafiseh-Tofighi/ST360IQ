import os
import torch
import torchvision
import numpy as np
import cv2
from tqdm import tqdm

from config import Config
from src.resnet50 import resnet50_backbone
from src.ViT_model import IQARegression
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error
import statistics

def test_epoch(check_path, test_scene, src_dir):
    config = Config({
        'gpu_id': 0,                                           # specify gpu number to use
        'checkpoint': check_path,                              # weights of trained model

        'batch_size': 1,
        'tan_data_path':src_dir,

        # ViT structure
        'n_enc_seq': [16,16],                   # input feature map dimension (N = H*W) from backbone
        'n_layer': 14,                          # number of encoder layers
        'd_hidn': 384,                          # input channel of encoder (input: C x N)
        'i_pad': 0,
        'd_ff': 384,                            # feed forward hidden layer dimension
        'd_MLP_head': 1152,                     # hidden layer of final MLP
        'n_head': 6,                            # number of head (in multi-head attention)
        'd_head': 384,                          # channel of each head -> same as d_hidn
        'dropout': 0.1,                         # dropout ratio
        'emb_dropout': 0.1,                     # dropout ratio of input embedding
        'layer_norm_epsilon': 1e-12,
        'n_output': 1,                          # dimension of output
        'patch_size':32,
    })


    # device setting
    config.device = torch.device('cuda:%s' % config.gpu_id if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print('Using GPU %s' % config.gpu_id)
    else:
        print('Using CPU')

    pred_total, target_total = test_model(config, test_scene)

    # compute correlation coefficient
    srcc, _ = spearmanr(pred_total, target_total)
    plcc, _ = pearsonr(pred_total, target_total)
    rmse = mean_squared_error(target_total,pred_total, squared=False)

    print('[test] / /SROCC:%4f / PLCC:%4f / RMSE:%4f' % 
                                                        (srcc, plcc, rmse))
    return {'plcc':plcc, 'srcc': srcc, 'rmse': rmse}


def test_model(config, test_scene):

    # input normalize
    class Normalize(object):
        def __init__(self, mean, var):
            self.mean = mean
            self.var = var
        def __call__(self, sample):
            sample[:, :, 0] = (sample[:, :, 0] - self.mean[0]) / self.var[0]
            sample[:, :, 1] = (sample[:, :, 1] - self.mean[1]) / self.var[1]
            sample[:, :, 2] = (sample[:, :, 2] - self.mean[2]) / self.var[2]
            return sample

    # numpy array -> torch tensor
    class ToTensor(object):
        def __call__(self, sample):
            sample = np.transpose(sample, (2, 0, 1))
            sample = torch.from_numpy(sample)
            return sample

    # create model
    model_backbone = resnet50_backbone().to(config.device)
    model_transformer = IQARegression(config).to(config.device)

    # load weights
    checkpoint = torch.load(config.checkpoint)
    model_backbone.load_state_dict(checkpoint['model_backbone_state_dict'])
    model_transformer.load_state_dict(checkpoint['model_transformer_state_dict'])
    model_backbone.eval()
    model_transformer.eval()

    # input transform
    transforms = torchvision.transforms.Compose(
        [Normalize([0.485, 0.456, 0.406], 
        [0.229, 0.224, 0.225]), 
        ToTensor()]
        )

    # save results
    pred_total = []
    target_total = []

    # input mask (batch_size x len_sqe+1)
    mask_inputs = torch.ones(config.batch_size, (config.n_enc_seq[0]*config.n_enc_seq[1])+1).to(config.device)

    pred_dict={}
    target_dict={}

    for filename in tqdm(test_scene):
        
        img_dir = os.path.join(config.tan_data_path, filename)
        data=torch.load(img_dir)

        center=data['center']
        center=torch.reshape(torch.tensor(center), (config.batch_size,2))
        source_emb=data['source_emb']

        source_emb=torch.reshape(torch.tensor(source_emb), (config.batch_size,1))
        score=float(data['score'])

        d_img=data['d_img_org']

        h, w=config.n_enc_seq[0]*config.patch_size, config.n_enc_seq[1]*config.patch_size
        
        d_img=cv2.resize(d_img,(h,w))
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype('float32') / 255

        d_img = transforms(d_img)
        d_img= torch.tensor(d_img.to(config.device)).unsqueeze(0).float()
            
        feat_dis_org = model_backbone(d_img)

        pred = model_transformer(mask_inputs, feat_dis_org,center,source_emb)

        # Averaging scores for all tan images of a single ERP image
        if source_emb in pred_dict:
            pred_dict[source_emb].append(float(pred.item()))
        else:
            pred_dict[source_emb]=[]
            pred_dict[source_emb].append(float(pred.item()))


        if source_emb in target_dict:
            target_dict[source_emb].append(score)
        else:
            target_dict[source_emb]=[]
            target_dict[source_emb].append(score)
        
    for img in pred_dict:
        pred_total.append(statistics.mean(pred_dict[img]))
        target_total.append(statistics.mean(target_dict[img]))

    return pred_total, target_total

