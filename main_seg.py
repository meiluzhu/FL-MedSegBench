
import time
import torch
import numpy as np
import os
import copy
import gc
import pprint
import argparse
import warnings
from datasets import Data
from nodes import Node
from server_funct import Server_update
from client_funct_seg import Client_update
from utils import setup_seed, set_server_method, lr_scheduler, validate_seg, FedRDNTransform
from nodes import Seed_Averager_seg

warnings.filterwarnings('ignore')
np.set_printoptions(precision=7, suppress=True)


_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--iid', type=int, default=0,
                        help='set 1 for iid, and 0 for noniid (dir. sampling)')
    parser.add_argument('--batchsize', type=int, default=128, 
                        help="batchsize")
    parser.add_argument('--num_classes', type=int, default=8, 
                        help="num_classes")
    
    # System
    parser.add_argument('--device', type=str, default='0',
                        help="cuda device: {cuda, cpu}")
    parser.add_argument('--node_num', type=int, default=20, 
                        help="Number of nodes") 
    parser.add_argument('--T', type=int, default=200, 
                        help="Number of communication rounds")
    parser.add_argument('--E', type=int, default=3, 
                        help="Number of local epochs: E")
    parser.add_argument('--dataset', type=str, default='OCT',
                        help="Type of dataset") 
    parser.add_argument('--data_path', type=str, default='./',
                        help="data_path") 
    parser.add_argument('--local_model', type=str, default='CNN',
                        help='Type of local model: {CNN, ResNet20, ResNet18}')
    parser.add_argument('--exp_name', type=str, default='FirstTable',
                        help="experiment name")

    # Server function
    parser.add_argument('--server_method', type=str, default='fedavg',
                        help="FedAvg, or others")
    # Client function
    parser.add_argument('--client_method', type=str, default='local_train',
                        help="client method")
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help="optimizer: {sgd, adam}")
    parser.add_argument('--lr', type=float, default=0.04,  
                        help='learning rate')
    parser.add_argument('--local_wd_rate', type=float, default=5e-4,
                        help='clients local wd rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')

    parser.add_argument('--method', type=str, default='FedAvg',
                        help="method")
 
    parser.add_argument('--mu', type=float, default=0.01,
                    help="FedProx mu")

    parser.add_argument('--loss', type=str, default='bce',
                    help="facal")
    
    ####personalization step
    parser.add_argument('--is_personalize', type=int, default=0, 
                        help="personalization")
    parser.add_argument('--lr_per', type=float, default=0.0001,
                    help="lr for personalization")
    parser.add_argument('--model_path', type=str, default='./',
                        help="model_path")

    #FedLWS
    parser.add_argument('--beta', type=float, default=0.03, 
                        help="beta")
    parser.add_argument('--min_tau', type=float, default=0.01,
                        help="min of tau")
    parser.add_argument('--max_tau', type=float, default=0.2,
                        help="max of tau")

    ### for FedAWA
    parser.add_argument('--server_optimizer', type=str, default='adam',
                        help="server_optimizer")
    parser.add_argument('--server_epochs', type=int, default=1,
                        help="optimizer epochs on server")
    parser.add_argument('--reg_distance', type=str, default='cos',
                        help="cos or euc")

    args = parser.parse_args()
    

    if args.dataset == 'Fundus': 
        random_seeds = [0, 1, 2]#
        #args.client_names = ['ARIA', 'ChaseDB', 'DR-Hagis', 'DRIVE', 'HRF', 'IOSTAR','LES-AV','ORVS']
        args.client_names = ['ChaseDB', 'DR-Hagis', 'DRIVE', 'HRF','LES-AV','ORVS']
        args.node_num = len(args.client_names)
        args.select_ratio = 1.0
        args.stepsize = args.T//4
        
    if args.dataset == 'Prostate': 
        random_seeds = [0, 1, 2]#
        #BIDMC  BMC  HK  I2CVB  RUNMC  UCL
        args.client_names = ['BIDMC', 'BMC', 'HK', 'I2CVB','RUNMC','UCL']
        args.node_num = len(args.client_names)
        args.select_ratio = 1.0
        args.num_classes = 1
        args.stepsize = args.T//4
        
    if args.dataset == 'Meibomian_Gland': 
        random_seeds = [0, 1, 2]#
        args.client_names = ['MGD1k', 'CAMG']
        args.node_num = len(args.client_names)
        args.select_ratio = 1.0
        args.stepsize = args.T//4
        
    if args.dataset == 'Pancreas': 
        random_seeds = [0, 1, 2]#
        args.client_names = ['AHN', 'NYU', 'MCA', 'NWU', 'MCF']
        args.node_num = len(args.client_names)
        args.select_ratio = 1.0
        args.stepsize = args.T//4
        args.in_dim = 1
        
    if args.dataset == 'Polyp': 
        random_seeds = [0, 1, 2]#
        args.client_names = ['CVC-300', 'CVC-ClinicDB', 'CVC-ColonDB', 'EndoTect', 'ETISLaribPolypDB'] # , 'Kvasir-SEG'
        args.node_num = len(args.client_names)
        args.select_ratio = 1.0
        args.stepsize = args.T//4
        
    if args.dataset == 'Pathology_COSAS2024': 
        random_seeds = [0, 1, 2]#
        args.client_names = ['3d-1000', 'kfbio-400', 'teksqray-600p']
        args.node_num = len(args.client_names)
        args.select_ratio = 1.0
        args.stepsize = args.T//4
        
    if args.dataset == 'FL_Breast_Ultrasound': 
        random_seeds = [0, 1, 2]#
        args.client_names = ['BUID', 'BUSI','BUS_UC', 'BUS_UCLM']
        args.node_num = len(args.client_names)
        args.select_ratio = 1.0
        args.stepsize = args.T//4
        
        
    if args.dataset == 'KiTS19': 
        random_seeds = [0, 1, 2]#
        args.client_names = ['client_0', 'client_1', 'client_2', 'client_3', 'client_4', 'client_5']
        args.node_num = len(args.client_names)
        args.select_ratio = 1.0
        args.stepsize = args.T//4
        args.num_classes = 3
        
    if args.dataset == 'MMS': 
        random_seeds = [0, 1, 2]#
        args.client_names = ['site1', 'site2', 'site3', 'site4', 'site5']
        args.node_num = len(args.client_names)
        args.select_ratio = 1.0
        args.stepsize = args.T//4
        args.num_classes = 4
        args.in_dim = 1
        
    if args.dataset == 'FeTS2022': 
        random_seeds = [0, 1, 2]#
        args.client_names = ['site1', 'site2', 'site3', 'site10', 'site22', 'site24', 'site25', 'site26', 'site28']
        args.node_num = len(args.client_names)
        args.select_ratio = 1.0
        args.stepsize = args.T//4
        args.num_classes = 4
        args.in_dim = 4
        
    if args.method == 'FedProx':
        args.mu = 0.01
        
    if args.method in ['MOON', 'FedDYN'] :
        args.mu = 0.01
    elif args.method in ['FedProx', 'Ditto']:
        args.mu = 0.001
    
    lr = args.lr 
    
    best_averagers = []

    for i, client_name in enumerate(args.client_names):   
        best_averagers.append(Seed_Averager_seg(i, client_name))
        
    models_save_path = './results/models'
    if not os.path.exists(models_save_path):
        os.makedirs(models_save_path)
        
    logs_save_path = './results/logs'
    if not os.path.exists(logs_save_path):
        os.makedirs(logs_save_path)
        
    for random_seed in random_seeds:
        best_val_dice = [0 for _ in args.client_names]
        gc.collect()
        torch.cuda.empty_cache()
        args.random_seed = random_seed
        args.lr = lr
        print('starting run seed', args.random_seed)
        setup_seed(random_seed)
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        print('The starting time ：{}'.format(now), flush=True)
        args = set_server_method(args)
        pprint(vars(args))
    
        select_list_recorder = [[i for i in range(args.node_num)] for _ in range(args.T)]

        setting_name =  args.method + '_' + args.dataset + '_' + args.local_model + '_nodenum' + str(args.node_num) +'_E'+ str(args.E) \
        + '_seed' + str(args.random_seed)
    
        root_path = './'
        output_path = 'results/'
        if not os.path.exists(os.path.join(root_path, output_path)):
            os.makedirs(os.path.join(root_path, output_path))
        #os.environ['CUDA_VISIBLE_DEVICES'] = args.device
        data = Data(args)

        sample_size = []
        for i in range(args.node_num): 
            sample_size.append(len(data.train_loaders[i]))
        size_weights = [i/sum(sample_size) for i in sample_size]
        
        #size_weights = [1.0/args.node_num for i in range(args.node_num)]
        print('size-based weights',size_weights, flush=True)

        central_node = Node(args,-1 , 'Server', train_loader = None, val_loader=None, test_loader=None)
        # initialize the client nodes
        client_nodes = {}
        for i in range(args.node_num): 
            client_nodes[i] = Node(args, i, args.client_names[i] , train_loader=data.train_loaders[i], val_loader=data.val_loaders[i], test_loader=data.test_loaders[i]) 
            client_nodes[i].model.load_state_dict(copy.deepcopy(central_node.model.state_dict()))
            #########################Scaffold##############################
            if args.method == 'Scaffold':
                client_nodes[i].control = copy.deepcopy(central_node.control)
                client_nodes[i].delta_control = copy.deepcopy(central_node.delta_control)
                client_nodes[i].delta_y = copy.deepcopy(central_node.delta_y)
    
        if args.method == 'FedRDN':
            global_stats = [] 
            for i in range(args.node_num): 
                global_stats.append(client_nodes[i].local_stats) 
            for i in range(args.node_num):
                client_nodes[i].FedRDNTransform_train = FedRDNTransform(
                    local_stats=client_nodes[i].local_stats,
                    global_stats=global_stats,
                    mode='train')
                client_nodes[i].FedRDNTransform_test = FedRDNTransform(
                    local_stats=client_nodes[i].local_stats,
                    global_stats=global_stats,
                    mode='test')
    
        print(setting_name, flush=True)
        
        training_logs = []
        for rounds in range(0, args.T):
            
            round_logs = {}
            round_logs['rounds'] = rounds
            
            print('===============Stage 1 The {:d}-th round==============='.format(rounds + 1), flush=True)
            
            if args.dataset == 'Fundus' or args.dataset == 'Meibomian_Gland': #args.dataset == 'FL_Skin' or args.dataset == 'FL_Breast_Ultrasound' or args.dataset == 'Pathology_COSAS2024' args.dataset == 'FL_Ultrasound'
                args = lr_scheduler(rounds, client_nodes, args)
            
            # Client selection
            select_list = select_list_recorder[rounds]
            # Local update
            client_nodes, client_train_losses, client_train_dices = Client_update(args, client_nodes, central_node, select_list)
            round_logs['client_train_losses']=client_train_losses
            round_logs['client_train_dices']=client_train_dices
            
            for i in select_list: 
                print('Train {:<12.12}, loss:{:.5f}, Dice:{:.5f}'.format(args.client_names[i],client_train_losses[i], client_train_dices[i] ), flush=True)
            print()
            
            if args.method == 'MOON':
                for i in select_list:
                    client_nodes[i].pre_model = copy.deepcopy(client_nodes[i].model)
            
            # Server aggregation
            central_node, client_nodes = Server_update(args, central_node, client_nodes, select_list, size_weights, rounds)
            
            client_val_losses = []
            client_val_dices = []
            for i in select_list:
                val_loss, val_dice, val_iou, val_sensitivity, val_specificity  = validate_seg(args, client_nodes[i], client_nodes[i].val_loader)
                client_val_losses.append(val_loss)
                client_val_dices.append(val_dice)
                print('Val   {:<12.12}, loss:{:.5f}, Dice:{:.3f}, IOU:{:.3f}, Sen:{:.3f}, Spe:{:.3f}'.format(args.client_names[i],val_loss, val_dice, val_iou, val_sensitivity, val_specificity), flush=True)
                
                if val_dice>best_val_dice[i] and rounds>args.T//2:
                    best_val_dice[i] = val_dice
                    if args.dataset != 'Prostate3D':
                        test_loss, test_dice, test_iou, test_sensitivity, test_specificity  = validate_seg(args, client_nodes[i], client_nodes[i].test_loader)
                        client_nodes[i].recorder.update(rounds, test_loss, test_dice, test_iou, test_sensitivity, test_specificity)
                    else:
                        client_nodes[i].recorder.update(rounds, val_loss, val_dice, val_iou, val_sensitivity, val_specificity)

                    #torch.save(client_nodes[i].model.state_dict(), os.path.join(save_path,args.client_names[i]+'_finalmodel_E'+str(args.E)+'_seed'+str(args.random_seed)+'.pth'))
            round_logs['client_val_losses']=client_val_losses
            round_logs['client_val_dices']=client_val_dices
            training_logs.append(round_logs)
            
            print()
            
            for i in select_list:
                client_nodes[i].recorder.log(is_log = True)
            print()
 
        for i in select_list: 
            loss, dice, iou, sensitivity, specificity, epoch = client_nodes[i].recorder.log(is_log = False)
            best_averagers[i].update(dice, iou, sensitivity, specificity)

        torch.save(training_logs, os.path.join(logs_save_path, setting_name+'_log.pt'))
        
        end = time.strftime("%Y-%m-%d %H:%M:%S")
        print('The ending time ：{}'.format(end))
    
    print('==========================best=================================')
    for i in select_list:
        best_averagers[i].log(details = True)
    print()
    
    print('==========================best=================================')
    for i in select_list:
        best_averagers[i].log(is_log = True)
    print()
    
    print('==========================average=================================')
    total_num = 0.
    total_dice = 0.
    total_iou = 0.
    total_sensitivity = 0.
    total_specificity = 0.
    
    dices = []
    ious = []
    sensitivities = []
    specificities = []
    
    for i in select_list:
        mean_dice, _, mean_iou, _, mean_sensitivity, _, mean_specificity, _ = best_averagers[i].log(is_log = False, details = False)
        if args.dataset != 'Prostate3D':
            local_num = client_nodes[i].test_loader.dataset.__len__()
        else:
            local_num = client_nodes[i].val_loader.dataset.__len__()
        total_num += local_num
        total_dice += mean_dice*local_num
        total_iou += mean_iou*local_num
        total_sensitivity += mean_sensitivity*local_num
        total_specificity += mean_specificity*local_num
        
        dices.append(mean_dice)
        ious.append(mean_iou)
        sensitivities.append(mean_sensitivity)
        specificities.append(mean_specificity)
    print('Test Average, Dice:{:.3f}, IOU:{:.3f}, Sen:{:.3f}, Spe:{:.3f}'.format(total_dice/total_num,total_iou/total_num,total_sensitivity/total_num,total_specificity/total_num), flush=True)

    mean_dice = np.mean(dices)
    std_dice = np.std(dices)
    
    mean_iou = np.mean(ious)
    std_iou = np.std(ious)
    
    mean_sensitivity = np.mean(sensitivities)
    std_sensitivity = np.std(sensitivities)
    
    mean_specificity = np.mean(specificities)
    std_specificity = np.std(specificities)
    
    print('Test Average, Dice:{:.3f}[{:.3f}], IOU:{:.3f}[{:.3f}], Sen:{:.3f}[{:.3f}], Spe:{:.3f}[{:.3f}]'.format(mean_dice, std_dice, mean_iou, std_iou, mean_sensitivity, std_sensitivity, mean_specificity, std_specificity), flush=True)
    
    print()
    
    
    
    
    