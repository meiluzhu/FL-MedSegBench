
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
from client_funct import Client_update
from utils import setup_seed, set_server_method, lr_scheduler, validate
from nodes import Seed_Averager

warnings.filterwarnings('ignore')
np.set_printoptions(precision=7, suppress=True)

def generate_matchlist(node_num, ratio = 0.5):
    candidate_list = [i for i in range(node_num)]
    select_num = int(ratio * node_num)
    match_list = np.random.choice(candidate_list, select_num, replace = False).tolist()
    return match_list

_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)



def dynamic_threshold(args, e):
    # args.Ek
    # args.T
    # args.alpha
    if e < args.Ek:
        args.percent_param = e/args.Ek * args.alpha
    else:
        args.percent_param = args.alpha
        
    print('percent_param:', args.percent_param)
        
    return args


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--iid', type=int, default=0,
                        help='set 1 for iid, and 0 for noniid (dir. sampling)')
    parser.add_argument('--batchsize', type=int, default=128, 
                        help="batchsize")
    parser.add_argument('--dirichlet_alpha', type=float, default=0.5, 
                    help="dirichlet_alpha")
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
    parser.add_argument('--select_ratio', type=float, default=1.0,
                    help="the ratio of client selection in each round")
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
 
    parser.add_argument('--percent', type=float, default=0.1,
                    help="percent")
    
    parser.add_argument('--mu', type=float, default=0.01,
                    help="FedProx mu")
    
    parser.add_argument('--percent_param', type=float, default=1.0,
                    help="percent_param")
    parser.add_argument('--Ek', type=int, default=0,
                    help="Ek")
    parser.add_argument('--alpha', type=float, default=0.5,
                    help="alpha")

    parser.add_argument('--server_epochs', type=int, default=100,
                    help="server_epochs")
    parser.add_argument('--server_interval', type=int, default=50,
                    help="server_interval")
    parser.add_argument('--server_optimizer', type=str, default='sgd',
                    help="server_optimizer")

    
    args = parser.parse_args()
    
    #Ensure that each client has samples
    if args.dataset == 'Kvasir': random_seeds = [1, 3, 6]
    if args.dataset == 'OCT': random_seeds = [0]
    if args.dataset == 'Digit': 
        random_seeds = [0, 1, 2]
        args.client_names = ['MNIST', 'MNIST_M', 'SVHN', 'SynthDigits', 'USPS']
        args.node_num = len(args.client_names)
        args.select_ratio = 1.0
        
        
    if args.method == 'FedProx':
        args.mu = 0.01
    
    lr = args.lr
    best_acc, best_recall, best_prec, best_f1, best_auc = [],[],[],[],[]
    last_acc, last_recall, last_prec, last_f1, last_auc = [],[],[],[],[]
    
    best_averagers = []
    last_averagers = []
    for i, client_name in enumerate(args.client_names):   
        best_averagers.append(Seed_Averager(i, client_name))
        last_averagers.append(Seed_Averager(i, client_name))
        
    for random_seed in random_seeds:
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
    
        if args.select_ratio == 1.0:
            select_list_recorder = [[i for i in range(args.node_num)] for _ in range(args.T)]
        else:
            select_list_recorder = [generate_matchlist(args.node_num, args.select_ratio) for _ in range(args.T)]
  
        setting_name =  args.exp_name + '_' + args.dataset + '_' + args.local_model + '_nodenum' + str(args.node_num) + '_dir' + str(args.dirichlet_alpha) +'_E'+ str(args.E)  + '_C' + str(args.select_ratio) \
        + '_' + args.server_method + '_' + args.client_method + '_seed' + str(args.random_seed)
    
        root_path = './'
        output_path = 'results/'
        if not os.path.exists(os.path.join(root_path, output_path)):
            os.makedirs(os.path.join(root_path, output_path))
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
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
            train_loaders_trans = None
            if args.method == 'FedFisher':
                train_loaders_trans = data.train_loaders_trans[i]
            client_nodes[i] = Node(args, i, args.client_names[i] , train_loader=data.train_loaders[i], val_loader=None, test_loader=data.test_loaders[i], train_loaders_trans=train_loaders_trans) 
            client_nodes[i].model.load_state_dict(copy.deepcopy(central_node.model.state_dict()))

        print(setting_name, flush=True)
        for rounds in range(0, args.T):
            
            print('===============Stage 1 The {:d}-th round==============='.format(rounds + 1), flush=True)
            
            #lr_scheduler(rounds, client_nodes, args)
            if args.method == 'FedFisher':
                args = dynamic_threshold(args, rounds)

            # Client selection
            select_list = select_list_recorder[rounds]
            # Local update
            client_nodes, client_train_losses, client_train_accs = Client_update(args, client_nodes, central_node, select_list)
            for i in select_list: 
                print('Train {:<12}, loss:{:.5f}, ACC:{:.3f}'.format(args.client_names[i],client_train_losses[i], client_train_accs[i] ), flush=True)
            print()
            
            # Server aggregation
            central_node, client_nodes = Server_update(args, central_node, client_nodes, select_list, size_weights, rounds)
            
            for i in select_list:
                val_loss, val_acc, val_recall, val_prec, val_f1, val_auc  = validate(args, client_nodes[i], client_nodes[i].test_loader)
                print('Test  {:<12}, loss:{:.5f}, ACC:{:.3f}, Recall:{:.3f}, Prec:{:.3f}, F1:{:.3f}, AUC:{:.3f}'.format(args.client_names[i],val_loss, val_acc, val_recall, val_prec, val_f1, val_auc), flush=True)
                client_nodes[i].maxer.update(rounds, val_loss, val_acc, val_recall, val_prec, val_f1, val_auc)
                # record last 5 epoch
                if rounds>=args.T-5:
                    client_nodes[i].averager.update(val_loss, val_acc, val_recall, val_prec, val_f1, val_auc)
            print()
            # print max for each epoch
            for i in select_list: 
                loss, acc, recall, prec, f1, auc, epoch = client_nodes[i].maxer.log(is_log = True)
            print()
            
            '''
            if val_acc+val_f1>best_val_acc:
                best_val_acc = val_acc+val_f1
                torch.save(central_node.model.state_dict(), os.path.join(root_path, output_path, setting_name+'_finalmodel.pth'))
            '''
        for i in select_list: 
            loss, acc, recall, prec, f1, auc, epoch = client_nodes[i].maxer.log(is_log = False)
            best_averagers[i].update(acc, recall, prec, f1, auc)
            
        print('Last 5 epochs:')
        for i in select_list: 
            loss, acc, recall, prec, f1, auc = client_nodes[i].averager.log(is_log = True)
            last_averagers[i].update(acc, recall, prec, f1, auc)
        print()

        end = time.strftime("%Y-%m-%d %H:%M:%S")
        print('The ending time ：{}'.format(end))
    
    print('==========================best=================================')
    for i in select_list:
        best_averagers[i].log(details = True)
    print()
    
    print('==========================last=================================')
    for i in select_list:
        last_averagers[i].log(details = True)
    print()
    
    
    print('==========================best=================================')
    for i in select_list:
        best_averagers[i].log(is_log = True)
    print()
    
    print('==========================last=================================')
    for i in select_list:
        last_averagers[i].log(is_log = True)
    print()
    
    '''
    print('==========================best=================================')
    print('Best test acc:', best_acc)
    print('Best test acc mean: {:.5f}'.format(np.mean(best_acc)),'Best test acc std: {:.5f}'.format(np.std(best_acc)) )

    print('Best test recall:', best_recall)
    print('Best test recall mean: {:.5f}'.format(np.mean(best_recall)),'Best test recall std: {:.5f}'.format(np.std(best_recall)) )

    print('Best test prec:', best_prec)
    print('Best test prec mean: {:.5f}'.format(np.mean(best_prec)),'Best test prec std: {:.5f}'.format(np.std(best_prec)) )

    print('Best test f1:', best_f1)
    print('Best test f1 mean: {:.5f}'.format(np.mean(best_f1)),'Best test f1 std: {:.5f}'.format(np.std(best_f1)) )

    print('Best test auc:', best_auc)
    print('Best test auc mean: {:.5f}'.format(np.mean(best_auc)),'Best test auc std: {:.5f}'.format(np.std(best_auc)) )
    print('===========================================================')
    
        
    print('==========================last=================================')
    print('Best test acc:', last_acc)
    print('Best test acc mean: {:.5f}'.format(np.mean(last_acc)),'Best test acc std: {:.5f}'.format(np.std(last_acc)) )

    print('Best test recall:', last_recall)
    print('Best test recall mean: {:.5f}'.format(np.mean(last_recall)),'Best test recall std: {:.5f}'.format(np.std(last_recall)) )

    print('Best test prec:', last_prec)
    print('Best test prec mean: {:.5f}'.format(np.mean(last_prec)),'Best test prec std: {:.5f}'.format(np.std(last_prec)) )

    print('Best test f1:', last_f1)
    print('Best test f1 mean: {:.5f}'.format(np.mean(last_f1)),'Best test f1 std: {:.5f}'.format(np.std(last_f1)) )

    print('Best test auc:', last_auc)
    print('Best test auc mean: {:.5f}'.format(np.mean(last_auc)),'Best test auc std: {:.5f}'.format(np.std(last_auc)) )
    print('===========================================================')
    '''
    
    
    
    
    
    