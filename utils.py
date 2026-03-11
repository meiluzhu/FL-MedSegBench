
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import random
from torch.backends import cudnn
import math
from pyhessian import hessian
from torch.optim import Optimizer
from models_dict import densenet, resnet, cnn, UNet
import copy

from loss import DiceBCELoss, BinaryDistillationLoss, DiceCELoss

##############################################################################
# Tools
##############################################################################

def set_server_method(args):

    if args.method == 'FedAvg':
        args.client_method = 'local_train'
        args.server_method = 'fedavg'
    elif args.method == 'FedBN':
        args.client_method = 'local_train'
        args.server_method = 'fedbn'
    elif args.method == 'FedProx':
        args.client_method = 'fedprox'
        args.server_method = 'fedavg'
    elif args.method == 'SioBN':
        args.client_method = 'local_train'
        args.server_method = 'siobn'
    elif args.method == 'SingleSet':
        args.client_method = 'local_train'
        args.server_method = 'singleset'
    elif args.method == 'FedALA':
        args.client_method = 'fedala'
        args.server_method = 'fedavg'
    elif args.method == 'FedETF':
        args.client_method = 'fedetf'
        args.server_method = 'fedavg'
    elif args.method == 'FedRoD':
        args.client_method = 'fedrod'
        args.server_method = 'fedavg'
    elif args.method == 'FedDYN':
        args.client_method = 'feddyn'
        args.server_method = 'feddyn'
    elif args.method == 'Scaffold':
        args.client_method = 'scaffold'
        args.server_method = 'scaffold'
    elif args.method == 'Ditto':
        args.client_method = 'ditto'
        args.server_method = 'fedavg'
    elif args.method == 'MOON':
        args.client_method = 'moon'
        args.server_method = 'fedavg'
    elif args.method == 'FedNova':
        args.client_method = 'local_train'
        args.server_method = 'fednova'
    elif args.method == 'FedPer':
        args.client_method = 'local_train'
        args.server_method = 'fedper'
    elif args.method == 'PN':
        args.client_method = 'local_train'
        args.server_method = 'fedavg'
    elif args.method == 'FedRDN':
        args.client_method = 'fedrdn'
        args.server_method = 'fedavg'
    elif args.method == 'FedLWS':
        args.client_method = 'local_train'
        args.server_method = 'fedlws'
    elif args.method == 'FedAWA':
        args.client_method = 'local_train'
        args.server_method = 'fedawa'
    else:
        assert False

    return args


class Model(nn.Module):
    """For classification problem"""

    def __init__(self, config):
        super().__init__()
        self.config = config

    def get_params(self):
        return self.state_dict()

    def get_gradients(self, dataloader):
        raise NotImplementedError


def set_params(model, model_state_dict, exclude_keys=set()):
    """
        Reference: Be careful with the state_dict[key].
        https://discuss.pytorch.org/t/how-to-copy-a-modified-state-dict-into-a-models-state-dict/64828/4.
    """
    with torch.no_grad():
        for key in model_state_dict.keys():
            if key not in exclude_keys:
                model.state_dict()[key].copy_(model_state_dict[key])
    return model

def freeze_layers(model, layers_to_freeze):
    for name, p in model.named_parameters():
        try:
            if name in layers_to_freeze:
                p.requires_grad = False
            else:
                p.requires_grad = True
        except:
            pass
    return model

class ModelWrapper(Model):
    def __init__(self, base, head, config):
        """
            head and base should be nn.module
        """
        super(ModelWrapper, self).__init__(config)

        self.base = base
        self.head = head

    def forward(self, x, return_embedding):
        feature_embedding = self.base(x)
        out = self.head(feature_embedding)
        if return_embedding:
            return feature_embedding, out
        else:
            return out


class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def value(self):
        return self.total / float(self.steps)

def softmax_fuct(lrs):
    '''
    lrs is dict as {0:3, 1:3, 2:4}
    '''
    exp_cache = []
    softmax_lrs = {}
    for i in range(len(lrs)):
        exp_cache.append(math.exp(lrs[i]))
    
    for i in range(len(lrs)):
        softmax_lrs[i] = exp_cache[i]/sum(exp_cache)
    
    return softmax_lrs

def cos(x, y):
    fuct = nn.CosineSimilarity(dim=0)
    result = fuct(x, y)
    result = result.detach().cpu().numpy().tolist()
    return result

def get_cosGrad_matrix(gradients):
    client_num = len(gradients)
    matrix = [[0.0 for _ in range(client_num)] for _ in range(client_num)]

    for i in range(client_num):
        for j in range(client_num):
            if matrix[j][i] != 0.0:
                matrix[i][j] = matrix[j][i]
            else:
                matrix[i][j] = cos(gradients[i], gradients[j])
    
    return matrix

def model_parameter_vector(args, model):
    param = [p.view(-1) for p in model.parameters()]
    # vector = torch.concat(param, dim=0)
    vector = torch.cat(param, dim=0)
    return vector

##############################################################################
# Initialization function
##############################################################################

def init_model(model_type, args):
    
    num_classes = args.num_classes
    if model_type == 'CNN':
        if args.dataset == 'cifar10':
            model = cnn.CNNCifar10()
        else:
            model = cnn.CNNCifar100()
    elif model_type == 'ResNet18':
        model = resnet.ResNet18(num_classes)
    elif model_type == 'ResNet20':
        model = resnet.ResNet20(num_classes)
    elif model_type == 'ResNet56':
        model = resnet.ResNet56(num_classes)
    elif model_type == 'ResNet110':
        model = resnet.ResNet110(num_classes)
    elif model_type == 'WRN56_2':
        model = resnet.WRN56_2(num_classes)
    elif model_type == 'WRN56_4':
        model = resnet.WRN56_4(num_classes)
    elif model_type == 'WRN56_8':
        model = resnet.WRN56_8(num_classes)
    elif model_type == 'DenseNet121':
        model = densenet.DenseNet121(num_classes)
    elif model_type == 'DenseNet169':
        model = densenet.DenseNet169(num_classes)
    elif model_type == 'DenseNet201':
        model = densenet.DenseNet201(num_classes)
    elif model_type == 'MLP':
        model = cnn.MLP()
    elif model_type == 'LeNet5':
        model = cnn.LeNet5() 
    elif model_type == 'DigitModel':
        model = cnn.DigitModel() 
    elif model_type == 'UNet':
        model = UNet.UNet()
    elif model_type == 'UNet2D':
        from models_dict.unet2d import Unet2D
        if args.method == 'PN':
            model = Unet2D(norm='pn', num_classes=args.num_classes)
        else:
            model = Unet2D(num_classes=args.num_classes)
    elif model_type == 'UNet3D':
        from models_dict.unet3d import UNet3D
        if args.method == 'PN':
            model = UNet3D(in_dim=args.in_dim, out_dim=args.num_classes, num_filters=16, norm='pn')
        else:
            model = UNet3D(in_dim=args.in_dim, out_dim=args.num_classes, num_filters=16, norm='bn3d')
    elif model_type == 'VNet':
        from models_dict.vnet import VNet
        model = VNet(num_classes=args.num_classes)
    elif model_type == 'SANet':
        from models_dict.sanet import SANet
        if args.method == 'PN':
            model = SANet(in_channels=args.in_dim, nclasses=args.num_classes, nf=16, relu='relu', up_mode='upconv', dropout_p=0,
              deep_supervision=False, norm = 'pn')
        else:
            model = SANet(in_channels=args.in_dim, nclasses=args.num_classes, nf=16, relu='relu', up_mode='upconv', dropout_p=0,
              deep_supervision=False)
        
    return model

def init_optimizer(num_id, model, args):

    if args.client_method == 'scaffold':
        optimizer = ScaffoldOptimizer(model.parameters(), lr=args.lr, weight_decay=args.local_wd_rate)
    else:
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.local_wd_rate)
        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.local_wd_rate)

    return optimizer

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True

##############################################################################
# Training function
##############################################################################

def generate_matchlist(client_node, ratio = 0.5):
    candidate_list = [i for i in range(len(client_node))]
    select_num = int(ratio * len(client_node))
    match_list = np.random.choice(candidate_list, select_num, replace = False).tolist()
    return match_list

def lr_scheduler(rounds, node_list, args):
    # learning rate scheduler for decaying
    if (rounds+1)%args.stepsize == 0:
        args.lr /= 2.0 #0.99
        for i in range(len(node_list)):
            node_list[i].args.lr = args.lr
            node_list[i].optimizer.param_groups[0]['lr'] = args.lr
    print('Learning rate={:.4f}'.format(args.lr))
    return args
    

class PerturbedGradientDescent(Optimizer):
    def __init__(self, params, lr=0.01, mu=0.0):
        if lr < 0.0:
            raise ValueError(f'Invalid learning rate: {lr}')

        default = dict(lr=lr, mu=mu)

        super().__init__(params, default)

    @torch.no_grad()
    def step(self, global_params):
        for group in self.param_groups:
            for p, g in zip(group['params'], global_params):
                # g = g.cuda()
                if p.grad != None:
                    d_p = p.grad.data + group['mu'] * (p.data - g.data)
                    p.data.add_(d_p, alpha=-group['lr'])

##############################################################################
# Validation function
##############################################################################
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

def compute_metrics(pre, gt): #D, H, W
    pred = pre.cpu().numpy()
    gt = gt.cpu().numpy()
    acc=accuracy_score(gt, pred)
    recall=recall_score(gt, pred, average='micro')
    prec = precision_score(gt, pred, average='macro')
    f1 = f1_score(gt, pred, average='macro')

    return acc, recall, prec, f1

def compute_auc(pre_scores, gt, num_classes = 8):

    pre_scores = pre_scores.cpu().numpy()
    gt = gt.cpu().numpy()
    
    gt_one_hot = np.eye(num_classes)[gt]
    auc_score = roc_auc_score(gt_one_hot, pre_scores)
    
    return auc_score
    

def validate(args, node, test_loader):
    '''
    Generally, 'validate' refers to the local datasets of clients and 'local' refers to the server's testset.
    '''
    node.model.eval() 
    with torch.no_grad():
        preds = []
        targets = []
        pred_scores = []
        loss = 0
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            output, feature = node.model(data)
            
            loss_local =  F.cross_entropy(output, target)
            loss = loss + loss_local.item()
            pred = output.argmax(dim=1)
            pred_scores.append(output.softmax(dim=1)) # B, C
            preds.append(pred)
            targets.append(target.view_as(pred))
            
        
        
        pred_scores = torch.cat(pred_scores)
        preds = torch.cat(preds)
        targets = torch.cat(targets)
        acc, recall, prec, f1 = compute_metrics(preds, targets)
        auc = compute_auc(pred_scores, targets, num_classes=args.num_classes)
        loss = loss/preds.shape[0]
        
    return loss, acc*100, recall*100, prec*100, f1*100, auc*100


def validate_seg(args, node, test_loader):
    '''
    Generally, 'validate' refers to the local datasets of clients and 'local' refers to the server's testset.
    '''
    node.model.eval() 

    if args.dataset == 'Digit':
        criterion = torch.nn.CrossEntropyLoss()
    if args.dataset == 'FeTS2022' or args.dataset == 'Fundus' or args.dataset == 'Prostate' or args.dataset == 'Meibomian_Gland' or args.dataset == 'Pancreas' or args.dataset == 'Polyp' or args.dataset == 'Pathology_COSAS2024' or args.dataset == 'FL_Ultrasound' or args.dataset == 'FL_Breast_Ultrasound' or args.dataset == 'FL_Skin' or args.dataset == 'KiTS19' or args.dataset == 'MMS':
        if args.loss == 'bce':
            criterion = torch.nn.BCEWithLogitsLoss()
        elif args.loss == 'dice_bce':
            criterion = DiceBCELoss()
        else:
            assert False
            
    with torch.no_grad():

        dices = []
        ious = []
        sensitivitys = []
        specificitys = []
        loss = 0
        num = 0
        if args.dataset == 'Pancreas' or args.dataset == 'MMS': #
            for idx, batch in enumerate(test_loader):
                
                data = batch['img']
                target = batch['mask']
                
                data, target = data.cuda(), target.cuda()
                
                if args.dataset == 'Pancreas':
                    patch_size=(32, 256, 256)
                if args.dataset == 'MMS':
                    patch_size=(16, 256, 256)
                
                output = sliding_window_inference_3d(args, node, data, patch_size=patch_size, overlap=0.5,)
                
                #print('output-target',output.shape,target.shape)
                #output-target torch.Size([1, 4, 12, 196, 240]) torch.Size([1, 1, 12, 196, 240])
                loss_local =  criterion(output, target)
                
                loss = loss + loss_local.item()
                
                dice = dice_fn(output, target)
                iou = iou_fn(output, target)
                sensitivity, specificity = sensitivity_specificity_fn(output, target)
                dices.append(dice.item())
                ious.append(iou.item())
                sensitivitys.append(sensitivity.item())
                specificitys.append(specificity.item())
                num += data.shape[0]
        else:
            for idx, batch in enumerate(test_loader):
                
                data = batch['img']
                target = batch['mask']
                
                data, target = data.cuda(), target.cuda()
                
                if args.method == 'FedRoD':
                    logit, feature = node.model(data)
                    logit_p = node.p_head(feature)
                    output = logit + logit_p
                elif args.method == 'Ditto':
                    output, feature = node.p_model(data)
                elif args.method == 'FedRDN':
                    data = node.FedRDNTransform_test(data)
                    output, _ = node.model(data)
                else:
                    output, _ = node.model(data)
                #print('output-target',output.shape,target.shape, np.unique(target.cpu().numpy()))
                loss_local =  criterion(output, target)
                
                loss = loss + loss_local.item()
                
                dice = dice_fn(output, target)
                iou = iou_fn(output, target)
                sensitivity, specificity = sensitivity_specificity_fn(output, target)
                dices.append(dice.item())
                ious.append(iou.item())
                sensitivitys.append(sensitivity.item())
                specificitys.append(specificity.item())
                num += data.shape[0]
            
        dice = sum(dices) / num
        iou = sum(ious) / num
        sensitivity = sum(sensitivitys) / num
        specificity = sum(specificitys) / num
        loss = loss / num
        
    return loss, dice*100, iou*100, sensitivity*100, specificity*100




def sliding_window_inference_3d(
    args,
    node,
    volume,           # [1, C, D, H, W] 或 [C, D, H, W]
    patch_size=(32, 256, 256),
    overlap=0.5,      # 重叠比例
    ):
    """
    3D sliding window inference
    """
    stride = [int(p * (1 - overlap)) for p in patch_size]
    volume_shape = volume.shape[-3:]  # D, H, W
    
    output_shape = (args.num_classes, *volume_shape)
    output = torch.zeros(output_shape).cuda()
    count = torch.zeros(output_shape).cuda()
    
    gaussian_weight = create_gaussian_weight(patch_size)
    gaussian_weight = gaussian_weight.cuda()
    
    for d in range(0, volume_shape[0], stride[0]):
        for h in range(0, volume_shape[1], stride[1]):
            for w in range(0, volume_shape[2], stride[2]):
                d_end = min(d + patch_size[0], volume_shape[0])
                h_end = min(h + patch_size[1], volume_shape[1])
                w_end = min(w + patch_size[2], volume_shape[2])
                
                d_start = d_end - patch_size[0]
                h_start = h_end - patch_size[1]
                w_start = w_end - patch_size[2]
                
                if d_start < 0: d_start = 0
                if h_start < 0: h_start = 0
                if w_start < 0: w_start = 0
                
                d_end = min(d_start + patch_size[0], volume_shape[0])
                h_end = min(h_start + patch_size[1], volume_shape[1])
                w_end = min(w_start + patch_size[2], volume_shape[2])
                
                patch = volume[...,
                              d_start:d_end,
                              h_start:h_end,
                              w_start:w_end]

                original_shape = patch.shape[-3:]
                if original_shape != patch_size:
                    pad_d = patch_size[0] - patch.shape[-3]
                    pad_h = patch_size[1] - patch.shape[-2]
                    pad_w = patch_size[2] - patch.shape[-1]
                    
                    patch = torch.nn.functional.pad(
                        patch,
                        (0, pad_w, 0, pad_h, 0, pad_d),
                        mode='constant'
                    )
                with torch.no_grad():
                    if args.method == 'FedRoD':
                        logit, feature = node.model(patch)
                        logit_p = node.p_head(feature)
                        pred_patch = logit + logit_p
                    elif args.method == 'Ditto':
                        pred_patch, feature = node.p_model(patch)
                    elif args.method == 'FedRDN':
                        data = node.FedRDNTransform_test(patch)
                        pred_patch, _ = node.model(data)
                    else:
                        pred_patch, _ = node.model(patch)
                    pred_patch = pred_patch.squeeze(0)

                if original_shape != patch_size:
                    pred_patch = pred_patch[..., :d_end-d_start,
                                               :h_end-h_start,
                                               :w_end-w_start]
                    
                weight = gaussian_weight[..., :d_end-d_start,
                                            :h_end-h_start,
                                            :w_end-w_start]

                output[..., d_start:d_end, h_start:h_end, w_start:w_end] += (pred_patch * weight)
                count[..., d_start:d_end, h_start:h_end, w_start:w_end] += weight

    output = output / (count + 1e-8)
    return output.unsqueeze(0)

def create_gaussian_weight(patch_size, sigma=0.125):
    center = [p // 2 for p in patch_size]
    meshgrid = torch.meshgrid(
        [torch.arange(p, dtype=torch.float32) for p in patch_size],
        indexing='ij'
    )
    
    weight = torch.ones(patch_size, dtype=torch.float32)
    for i, (grid, c) in enumerate(zip(meshgrid, center)):
        weight *= torch.exp(-((grid - c) ** 2) / (2 * (sigma * patch_size[i]) ** 2))
    
    return weight / weight.max()





def dice_fn(y_pred, y_true, smooth=1e-6):
    """
    PyTorch implementation of the Dice coefficient (supports batch computation)

    Args:
        y_pred: Predicted probability map (B, C, H, W) or (B, C, D, H, W)
        y_true: Ground truth labels (B, H, W) or (B, D, H, W)
        smooth: Smoothing factor to avoid division by zero

    Returns:
        dice: Dice coefficient
    """
    num_classes = y_pred.shape[1]
    if num_classes==1:
        y_pred = torch.sigmoid(y_pred) #(B, 1, H, W), (B, 1, D, H, W)
        y_pred = (y_pred > 0.5).float()  # (B, 1, H, W), (B, 1, D, H, W)
        y_pred = y_pred.squeeze(1) # (B, H, W), (B, D, H, W)
        if y_pred.dim() == 3:
            intersection = (y_pred * y_true).sum(dim=(1, 2))
            sum_true_pred = y_pred.sum(dim=(1, 2)) + y_true.sum(dim=(1, 2))
        if y_pred.dim() == 4:
            intersection = (y_pred * y_true).sum(dim=(1, 2, 3))
            sum_true_pred = y_pred.sum(dim=(1, 2, 3)) + y_true.sum(dim=(1, 2, 3))
    else:
        y_pred = torch.softmax(y_pred, dim = 1)#B,C,H,W    B,C,D,H,W   
        y_pred = torch.argmax(y_pred, dim = 1)#B,H,W     B,D,H,W
        if y_pred.dim() == 3:
            y_pred = F.one_hot(y_pred, num_classes).permute(0, 3, 1, 2).float() #B,C,H,W
            y_true = F.one_hot(y_true.long(), num_classes).permute(0, 3, 1, 2).float()#B,C,H,W
            intersection = (y_pred * y_true).sum(dim=(2, 3))
            sum_true_pred = y_pred.sum(dim=(2, 3)) + y_true.sum(dim=(2, 3))
        
        if y_pred.dim() == 4:
            y_pred = F.one_hot(y_pred, num_classes).permute(0, 4, 1, 2, 3).float()#B,C,D,H,W
            y_true = F.one_hot(y_true.long(), num_classes).permute(0, 4, 1, 2, 3).float()#B,C,D,H,W
            intersection = (y_pred * y_true).sum(dim=(2, 3, 4))
            sum_true_pred = y_pred.sum(dim=(2, 3, 4)) + y_true.sum(dim=(2, 3, 4))
        
    dice = (2.0 * intersection + smooth) / (sum_true_pred + smooth)
    dice = dice/num_classes
        
    return dice.sum()  



def dice_fn_v1(y_pred, y_true, smooth=1e-6, num_class=2):
    """
    PyTorch implementation of the Dice coefficient (supports batch computation)

    Args:
        y_pred: Predicted probability map (B, 1, H, W), values in range [0, 1]
        y_true: Ground truth labels (B, 1, H, W), values in {0, 1}
        smooth: Smoothing factor to avoid division by zero

    Returns:
        dice: Dice coefficient
    """
    y_pred = torch.sigmoid(y_pred)
    y_pred = (y_pred > 0.5).float().squeeze(dim=1)  
    all_dice = 0
    batch_size = y_true.shape[0]
    y_true = y_true.squeeze(dim=1)
    for i in range(num_class):
        each_pred = torch.zeros_like(y_pred)
        each_pred[y_pred==i] = 1

        each_gt = torch.zeros_like(y_true)
        each_gt[y_true==i] = 1            
        
        intersection = torch.sum((each_pred * each_gt).view(batch_size, -1), dim=1)
        
        union = each_pred.view(batch_size,-1).sum(1) + each_gt.view(batch_size,-1).sum(1)
        dice = (2. *  intersection )/ (union + 1e-5)
        
        all_dice += torch.sum(dice) # B, 1
    
    return all_dice * 1.0 / num_class



def iou_fn(y_pred, y_true):
    """
    Compute IoU for binary segmentation

    Args:
        y_pred: Predicted probability map (B, C, H, W) or (B, C, D, H, W)
        y_true: Ground truth labels (B, H, W) or (B, D, H, W)
        threshold: Binarization threshold

    Returns:
        iou: Intersection over Union
    """
    num_classes = y_pred.shape[1]
    
    if num_classes == 1:
        y_pred = torch.sigmoid(y_pred)#(B, 1, H, W), (B, 1, D, H, W)
        y_pred = (y_pred > 0.5).float()#(B, 1, H, W), (B, 1, D, H, W)
        y_pred = y_pred.squeeze(1) # (B, H, W), (B, D, H, W)
        y_true = y_true.float() # (B, H, W), (B, D, H, W)
        
        if y_pred.dim() == 3:
            intersection = (y_pred * y_true).sum(dim=(1, 2))
            union = (y_pred + y_true).sum(dim=(1, 2)) - intersection
        if y_pred.dim() == 4:
            intersection = (y_pred * y_true).sum(dim=(1, 2, 3))
            union = (y_pred + y_true).sum(dim=(1, 2, 3)) - intersection
    else:
        y_pred = torch.softmax(y_pred, dim = 1)#(B, C, H, W), (B, C, D, H, W)
        y_pred = torch.argmax(y_pred, dim = 1)#(B, H, W), (B, D, H, W)
        if y_pred.dim() == 3:
            y_pred = F.one_hot(y_pred, num_classes).permute(0, 3, 1, 2).float() #B,C,H,W
            y_true = F.one_hot(y_true.long(), num_classes).permute(0, 3, 1, 2).float() #B,C,H,W
            intersection = (y_pred * y_true).sum(dim=(2, 3))
            union = (y_pred + y_true).sum(dim=(2, 3)) - intersection
            
        if y_pred.dim() == 4:
            y_pred = F.one_hot(y_pred, num_classes).permute(0, 4, 1, 2, 3).float()
            y_true = F.one_hot(y_true.long(), num_classes).permute(0, 4, 1, 2, 3).float()
            intersection = (y_pred * y_true).sum(dim=(2, 3, 4))
            union = (y_pred + y_true).sum(dim=(2, 3, 4)) - intersection
            
    iou = (intersection + 1e-6) / (union + 1e-6)
    iou = iou/num_classes
        
    return iou.sum()





def iou_fn_v1(y_pred, y_true, num_class=2):
    """
    Compute IoU for binary segmentation

    Args:
        y_pred: Predicted probability map (B, 1, H, W), values in [0, 1]
        y_true: Ground truth labels (B, 1, H, W), values in {0, 1}
        threshold: Binarization threshold

    Returns:
        iou: Intersection over Union
    """
    y_pred = torch.sigmoid(y_pred)
    y_pred = (y_pred > 0.5).float().squeeze(dim=1)  
    y_true = y_true.float().squeeze(dim=1)
    all_iou = 0
    batch_size = y_true.shape[0]
    for i in range(num_class):
        each_pred = torch.zeros_like(y_pred)
        each_pred[y_pred==i] = 1

        each_gt = torch.zeros_like(y_true)
        each_gt[y_true==i] = 1            
        
        intersection = torch.sum((each_pred * each_gt).view(batch_size, -1), dim=1)
        union = each_pred.view(batch_size,-1).sum(1) + each_gt.view(batch_size,-1).sum(1)
        
        iou = (intersection + 1e-6) / (union + 1e-6)
        
        all_iou += torch.sum(iou)
        
    
    return all_iou * 1.0 / num_class



def sensitivity_specificity_fn(y_pred, y_true):
    """
    Compute Sensitivity and Specificity for binary classification

    Args:
        y_pred: Predicted probability map (B, C, H, W) or (B, C, D, H, W)
        y_true: Ground truth labels (B, H, W) or (B, D, H, W)
        threshold: Binarization threshold

    Returns:
        sensitivity, specificity
    """
    num_classes = y_pred.shape[1]
    if num_classes == 1:
        y_pred = torch.sigmoid(y_pred)#(B, 1, H, W), (B, 1, D, H, W)
        y_pred = (y_pred > 0.5).float()#(B, 1, H, W), (B, 1, D, H, W)
        y_pred = y_pred.squeeze(1) # (B, H, W), (B, D, H, W)
        y_true = y_true.float() # (B, H, W), (B, D, H, W)
        
        # 计算 TP, FN, TN, FP
        TP = torch.sum(y_pred * y_true)      # True Positive
        FN = torch.sum((1 - y_pred) * y_true) # False Negative
        TN = torch.sum((1 - y_pred) * (1 - y_true)) # True Negative
        FP = torch.sum(y_pred * (1 - y_true)) # False Positive
        
    else:
        y_pred = torch.softmax(y_pred, dim = 1)#(B, C, H, W), (B, C, D, H, W)
        y_pred = torch.argmax(y_pred, dim = 1)#(B, H, W), (B, D, H, W)
        y_true = y_true.long()#(B, H, W), (B, D, H, W)
        if y_pred.dim() == 3:
            y_pred = F.one_hot(y_pred, num_classes).permute(0, 3, 1, 2).float()#B,C,H,W
            y_true = F.one_hot(y_true, num_classes).permute(0, 3, 1, 2).float()#B,C,H,W
 
            # Cal TP, FN, TN, FP
            TP = torch.sum(y_pred * y_true, dim=(2, 3))      # True Positive
            FN = torch.sum((1 - y_pred) * y_true, dim=(2, 3)) # False Negative
            TN = torch.sum((1 - y_pred) * (1 - y_true), dim=(2, 3)) # True Negative
            FP = torch.sum(y_pred * (1 - y_true), dim=(2, 3)) # False Positive
        
        if y_pred.dim() == 4:
            y_pred = F.one_hot(y_pred, num_classes).permute(0, 4, 1, 2, 3).float()
            y_true = F.one_hot(y_true, num_classes).permute(0, 4, 1, 2, 3).float()
            
            # 计算 TP, FN, TN, FP
            TP = torch.sum(y_pred * y_true, dim=(2, 3, 4))      # True Positive
            FN = torch.sum((1 - y_pred) * y_true, dim=(2, 3, 4)) # False Negative
            TN = torch.sum((1 - y_pred) * (1 - y_true), dim=(2, 3, 4)) # True Negative
            FP = torch.sum(y_pred * (1 - y_true), dim=(2, 3, 4)) # False Positive
            
    sensitivity = TP / (TP + FN + 1e-6)
    sensitivity = sensitivity.sum()/num_classes
    specificity = TN / (TN + FP + 1e-6)
    specificity = specificity.sum()/num_classes

    return sensitivity, specificity


#########################Scaffold##############################
from torch.optim import Optimizer

class ScaffoldOptimizer(Optimizer):
    def __init__(self, params, lr, weight_decay):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(ScaffoldOptimizer, self).__init__(params, defaults)

    def step(self, server_controls, client_controls, closure=None):

        loss = None
        if closure is not None:
            loss = closure

        for group in self.param_groups:
            for p, c, ci in zip(group['params'], server_controls.values(), client_controls.values()):
                if p.grad is None:
                    continue
                dp = p.grad.data + c.data - ci.data
                p.data = p.data - dp.data * group['lr']

        return loss
    
class FedRDNTransform:
    """
    Federated Random Data Normalization Transform

    Used for data augmentation in federated learning with feature distribution skew.
    """
    
    def __init__(self, 
                 local_stats,
                 global_stats,
                 mode: str = 'train',
                 p: float = 1.0):
        """
        Initialize the FedRDN transform

        Args:
            local_stats: Local statistics (mean, std)
            global_stats: List of global statistics [(mean1, std1), (mean2, std2), ...]
            mode: 'train' or 'test'
            p: Probability of applying the transform
        """
        self.local_mean, self.local_std = local_stats
        self.global_means = [s[0] for s in global_stats]
        self.global_stds = [s[1] for s in global_stats]
        self.mode = mode
        self.p = p
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:

        if random.random() > self.p:
            return x
            
        if self.mode == 'train':
            idx = random.randint(0, len(self.global_means) - 1)
            mean = self.global_means[idx]
            std = self.global_stds[idx]
        else:
            mean = self.local_mean
            std = self.local_std
            
        if len(x) == 5:
            x_normalized = (x - mean.view(-1, 1, 1, 1)) / (std.view(-1, 1, 1, 1) + 1e-8)
        else:
            x_normalized = (x - mean.view(-1, 1, 1)) / (std.view(-1, 1, 1) + 1e-8)
        
        return x_normalized