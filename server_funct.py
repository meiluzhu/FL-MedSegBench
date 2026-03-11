import numpy as np
import torch
import torch.nn.functional as F
import os
import random
from torch.backends import cudnn
from random import sample
import math
import torch.optim as optim
import torch.nn as nn
import copy
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import init_model, freeze_layers, set_params
from datasets import TensorDataset
from torch.autograd import Variable

def receive_client_models(args, client_nodes, select_list, size_weights):
    client_params = []
    local_protos_list = {}

    for idx in select_list:
        client_params.append(copy.deepcopy(client_nodes[idx].model.state_dict()))

    agg_weights = [size_weights[idx] for idx in select_list]
    agg_weights = [w/sum(agg_weights) for w in agg_weights]

    return agg_weights, client_params, local_protos_list


def Server_update(args, central_node, client_nodes, select_list, size_agg_weights, epoch = 0):
    '''
    server update functions for baselines
    '''
    # receive the local models from clients
    #agg_weights, client_params, local_protos_list = receive_client_models(args, client_nodes, select_list, size_weights)

    agg_weights = [size_agg_weights[idx] for idx in select_list]
    agg_weights = [w/sum(agg_weights) for w in agg_weights]
    
    # update the global model
    if args.server_method == 'fedavg':
        #avg_global_param = fedavg(client_params, agg_weights)
        #central_node.model.load_state_dict(avg_global_param)
        for key in central_node.model.state_dict().keys():
            # num_batches_tracked is a non trainable LongTensor and
            # num_batches_tracked are the same for all clients for the given datasets
            if 'num_batches_tracked' in key:
                central_node.model.state_dict()[key].data.copy_(client_nodes[0].model.state_dict()[key])
            else:
                temp = torch.zeros_like(central_node.model.state_dict()[key])
                for i, client_idx in enumerate(select_list):
                    temp += agg_weights[i] * client_nodes[client_idx].model.state_dict()[key]
                central_node.model.state_dict()[key].data.copy_(temp)
                for i, client_idx in enumerate(select_list):
                    client_nodes[client_idx].model.state_dict()[key].data.copy_(central_node.model.state_dict()[key])

    elif args.server_method == 'fednova':
        fed_avg_freqs = agg_weights
        client_step = [(len(client_nodes[i].train_loader)) for i in select_list]
        tao_eff = 0.0
        for j in range(len(fed_avg_freqs)):
            tao_eff += fed_avg_freqs[j] * client_step[j]
        correct_term = 0.0
        for j in range(len(fed_avg_freqs)):
            correct_term += fed_avg_freqs[j] / client_step[j] * tao_eff
        
        # Model aggregation
        for key in central_node.model.state_dict().keys():
            # num_batches_tracked is a non trainable LongTensor and
            # num_batches_tracked are the same for all clients for the given datasets
            if 'num_batches_tracked' in key:
                central_node.model.state_dict()[key].data.copy_(client_nodes[0].model.state_dict()[key])
            else:
                #temp = torch.zeros_like(central_node.model.state_dict()[key])
                temp = (1.0 - correct_term)*central_node.model.state_dict()[key]
                for i, client_idx in enumerate(select_list):
                    temp += fed_avg_freqs[client_idx] / client_step[client_idx] * tao_eff * client_nodes[client_idx].model.state_dict()[key]
                central_node.model.state_dict()[key].data.copy_(temp)
                for i, client_idx in enumerate(select_list):
                    client_nodes[client_idx].model.state_dict()[key].data.copy_(central_node.model.state_dict()[key])
    elif args.server_method == 'fedper':
        
        if args.local_model == 'UNet2D':
            out_key = 'seg1'
        elif args.local_model == 'UNet3D':
            out_key = 'out'
        elif args.local_model == 'SANet':
            out_key = 'last'
        else:
            assert False

        if args.local_model != 'UNet':
            for key in central_node.model.state_dict().keys():
                if out_key not in key:
                    if 'num_batches_tracked' in key:
                        central_node.model.state_dict()[key].data.copy_(client_nodes[0].model.state_dict()[key])
                    else:
                        temp = torch.zeros_like(central_node.model.state_dict()[key], dtype=torch.float32)
                        for i, client_idx in enumerate(select_list):
                            temp += agg_weights[i] * client_nodes[client_idx].model.state_dict()[key]
                        central_node.model.state_dict()[key].data.copy_(temp)
                        for i, client_idx in enumerate(select_list):
                            client_nodes[client_idx].model.state_dict()[key].data.copy_(central_node.model.state_dict()[key])
        else:
            for key in central_node.model.state_dict().keys():
                if 'final' not in key:
                    if 'num_batches_tracked' in key:
                        central_node.model.state_dict()[key].data.copy_(client_nodes[0].model.state_dict()[key])
                    else:
                        temp = torch.zeros_like(central_node.model.state_dict()[key], dtype=torch.float32)
                        for i, client_idx in enumerate(select_list):
                            temp += agg_weights[i] * client_nodes[client_idx].model.state_dict()[key]
                        central_node.model.state_dict()[key].data.copy_(temp)
                        for i, client_idx in enumerate(select_list):
                            client_nodes[client_idx].model.state_dict()[key].data.copy_(central_node.model.state_dict()[key])

    elif args.server_method == 'feddyn':
        # update server's state
        uploaded_models = []
        for i in select_list:
            uploaded_models.append(copy.deepcopy(client_nodes[i].model))
    
        model_delta = copy.deepcopy(uploaded_models[0])
        for param in model_delta.parameters():
            param.data = torch.zeros_like(param.data)
    
        for idx, client_model in enumerate(uploaded_models):
            for server_param, client_param, delta_param in zip(central_node.model.parameters(), client_model.parameters(), model_delta.parameters()):
                delta_param.data += (client_param - server_param) * agg_weights[idx]
    
        for state_param, delta_param in zip(central_node.server_state.parameters(), model_delta.parameters()):
            state_param.data -= args.mu * delta_param
    
        # aggregation
        central_node.model = copy.deepcopy(uploaded_models[0])
        for param in central_node.model.parameters():
            param.data = torch.zeros_like(param.data)
            
        for idx, client_model in enumerate(uploaded_models):
            for server_param, client_param in zip(central_node.model.parameters(), client_model.parameters()):
                server_param.data += client_param.data.clone() * agg_weights[idx]
    
        for server_param, state_param in zip(central_node.model.parameters(), central_node.server_state.parameters()):
            server_param.data -= (1/args.mu) * state_param
            
        for key in central_node.model.state_dict().keys():
            if 'num_batches_tracked' in key:
                central_node.model.state_dict()[key].data.copy_(client_nodes[0].model.state_dict()[key])
            else:
                for i, client_idx in enumerate(select_list):
                    client_nodes[client_idx].model.state_dict()[key].data.copy_(central_node.model.state_dict()[key])

    elif args.server_method == 'fedawa':
        
        global global_T_weight
        
        if epoch==0:
            global_T_weight = agg_weights

        def get_flat_weights(node):
            
            weights_flat = []
        
            for key in node.model.state_dict().keys():
                if ('num_batches_tracked' not in key) and ('running_mean' not in key) and ('running_var' not in key):
                    p = node.model.state_dict()[key]
                    weights_flat.append(p.clone().detach().reshape(-1))
            with torch.no_grad():
                weights_flat = torch.cat(weights_flat, 0)
            
            return weights_flat


        flat_w_list = []
        for _, client_idx in enumerate(select_list):
            c_weights_flat = get_flat_weights(client_nodes[client_idx])
            flat_w_list.append(c_weights_flat)
        local_param_list = torch.stack(flat_w_list)#C, N
        
        global_weights_flat = get_flat_weights(central_node)
        
        
        T_weights = torch.tensor(global_T_weight, dtype=torch.float32).cuda()
        T_weights = Variable(T_weights, requires_grad=True)
        
        if args.server_optimizer=='sgd':##########
            Attoptimizer = torch.optim.SGD([T_weights], lr=0.01, momentum=0.9, weight_decay=5e-4)
        elif args.server_optimizer=='adam':
            Attoptimizer = optim.Adam([T_weights], lr=0.001, betas=(0.5, 0.999))
        
        print("T_weights_before update:",torch.nn.functional.softmax(T_weights, dim=0).data.cpu())
        
        
        def _cost_matrix(x, y, dis, p=2):
            d_cosine = nn.CosineSimilarity(dim=-1, eps=1e-8)
        
            x_col = x.unsqueeze(-2)
            y_lin = y.unsqueeze(-3)
            if dis == 'cos':
                # print('cos_dis')
                C = 1-d_cosine(x_col, y_lin)
            elif dis == 'euc':
                # print('euc_dis')
                C= torch.mean((torch.abs(x_col - y_lin)) ** p, -1)
            return C
        
        for i in range(args.server_epochs):###########
            print("server weight update:",i)
            probability_train = torch.nn.functional.softmax(T_weights, dim=0)
            
            #########
            C = _cost_matrix(global_weights_flat.detach().unsqueeze(0), local_param_list.detach(), args.reg_distance)
            
            reg_loss = torch.sum(probability_train* C, dim=(-2, -1))
            print("reg_loss:",reg_loss.data.cpu())
            
            client_grad=local_param_list-global_weights_flat
            
            column_sum=torch.matmul(probability_train.unsqueeze(0),client_grad) #weighted sum
            l2_distance = torch.norm(client_grad.unsqueeze(0) - column_sum.unsqueeze(1), p=2, dim=2)
            
            print("L2_distance:",l2_distance.data.cpu())
            sim_loss=(torch.sum(probability_train*l2_distance, dim=(-2, -1)))
    
            print("Sim_loss:",sim_loss.data.cpu())
            
            Loss=sim_loss+reg_loss
            Attoptimizer.zero_grad()
            Loss.backward()
            Attoptimizer.step()
            print("step "+str(i)+" Loss:"+str(Loss))
            
        global_T_weight = T_weights.data
        print("T_weights_after update:",global_T_weight)
        print("probability_train_after update:",probability_train)

        for key in central_node.model.state_dict().keys():
            # num_batches_tracked is a non trainable LongTensor and
            # num_batches_tracked are the same for all clients for the given datasets
            if 'num_batches_tracked' in key:
                central_node.model.state_dict()[key].data.copy_(client_nodes[0].model.state_dict()[key])
            else:
                temp = torch.zeros_like(central_node.model.state_dict()[key])
                for i, client_idx in enumerate(select_list):
                    temp += probability_train[i] * client_nodes[client_idx].model.state_dict()[key]
                central_node.model.state_dict()[key].data.copy_(temp/sum(probability_train))
                for i, client_idx in enumerate(select_list):
                    client_nodes[client_idx].model.state_dict()[key].data.copy_(central_node.model.state_dict()[key])


    elif args.server_method == 'fedlws':
        #https://github.com/ChanglongShi/FedLWS/blob/main/server_funct.py
        client_params = []
        for idx in select_list:
            client_params.append(copy.deepcopy(client_nodes[idx].model.state_dict()))
        
        param=central_node.model.state_dict()
        global_params = copy.deepcopy(param)
        fedavg_global_params = copy.deepcopy(client_params[0])
        
        for name_param in client_params[0]:
            list_values_param = []
            for dict_local_params, num_local_data in zip(client_params,agg_weights):
                list_values_param.append(dict_local_params[name_param] * num_local_data)
            value_global_param = sum(list_values_param)# / sum(list_nums_local_data)
            fedavg_global_params[name_param] = value_global_param
        
        cur_w=[]
        last_w=[]
        l=torch.tensor([]).cuda()
        l_last=torch.tensor([]).cuda()
        for name_param in client_params[0]:
            # print(name_param)
            l=torch.cat((l,fedavg_global_params[name_param].reshape(-1)))
            l_last=torch.cat((l_last,global_params[name_param].reshape(-1)))
            # a=layer_cossim[i][name_param]
            if "bias" in name_param:
                cur_w.append(l)
                last_w.append(l_last)
                # print(name_param)
                # print(l.shape)
                # a=torch.tensor(0).cuda().float()
                l=torch.tensor([]).cuda()
                l_last=torch.tensor([]).cuda()
        clients_w=[]
        for i in range(len(client_params)):
            client_w=[]
            l_client=torch.tensor([]).cuda()
            for name_param in client_params[0]:
                l_client=torch.cat((l_client,client_params[i][name_param].reshape(-1)))
                # a=layer_cossim[i][name_param]
                if "bias" in name_param:
                    client_w.append(l_client)
                    # a=torch.tensor(0).cuda().float()
                    l_client=torch.tensor([]).cuda()
            clients_w.append(client_w)
            
            
        layer_gammas=[]
        ######### layer_tau ##############
        taus=[]
        for i in range(len(last_w)):
            # print(cur_w[i].shape)
            grad=torch.norm(cur_w[i]-last_w[i],p=2)
            layer_grad=[]
            for k in range(len(client_params)):
                layer_grad.append(clients_w[k][i]-last_w[i])
            global_grad=torch.mean(torch.stack(layer_grad),dim=0)
            l2_norms = [torch.norm(tensor - global_grad, p=2) for tensor in layer_grad]
            l2_norm_average = sum(l2_norms) / len(l2_norms)
            tau=args.beta*(l2_norm_average)
            tau = torch.clamp(tau, min=args.min_tau, max=args.max_tau)
            taus.append(tau)
            gamma=torch.norm(last_w[i],p=2)/(torch.norm(last_w[i],p=2)+tau*grad)
            layer_gammas.append(gamma)
        #print("taus:",taus)
        #print("layer_gammas:",layer_gammas)
        cur_layer=0
        for name_param in client_params[0]:
            # if name_param[-6:]=="weight":
            #     k+=1
            fedavg_global_params[name_param] = fedavg_global_params[name_param]*layer_gammas[cur_layer]#+fedavg_global_params[name_param]*(1-d0)
            if "bias" in name_param:
                cur_layer+=1

        ##added by meilu
        central_node.model.load_state_dict(fedavg_global_params)
        for key in central_node.model.state_dict().keys():
            if 'num_batches_tracked' in key:
                central_node.model.state_dict()[key].data.copy_(client_nodes[0].model.state_dict()[key])
            else:
                for i, client_idx in enumerate(select_list):
                    client_nodes[client_idx].model.state_dict()[key].data.copy_(central_node.model.state_dict()[key])


    elif args.server_method == 'scaffold':
        
        fedavg_global_params = {}
        for k, v in client_nodes[0].model.state_dict().items():
            list_values_param = []
            ck_list_values_param = []#########
            for j, num_local_data in zip(select_list, agg_weights):
                if k in client_nodes[0].delta_y.keys():
                    list_values_param.append(client_nodes[j].delta_y[k]/len(select_list))
                    ck_list_values_param.append(client_nodes[j].delta_control[k]/len(select_list))#########
                else:
                    list_values_param.append(client_nodes[j].model.state_dict()[k]/len(select_list))
            value_global_param = sum(list_values_param)# / sum(agg_weights)
            if k in client_nodes[0].delta_y.keys():
                fedavg_global_params[k] = value_global_param + central_node.model.state_dict()[k]
                ck_list_values_param = sum(ck_list_values_param)# / sum(agg_weights)#########
                central_node.control[k].data += ck_list_values_param * (len(select_list) / args.node_num)#########
            else:
                fedavg_global_params[k] = value_global_param
        central_node.model.load_state_dict(fedavg_global_params)

        for key in central_node.model.state_dict().keys():
            if 'num_batches_tracked' in key:
                central_node.model.state_dict()[key].data.copy_(client_nodes[0].model.state_dict()[key])
            else:
                for i, client_idx in enumerate(select_list):
                    client_nodes[client_idx].model.state_dict()[key].data.copy_(central_node.model.state_dict()[key])

    elif args.server_method == 'fedbn':
        if args.local_model == 'UNet2D' or args.local_model == 'SANet':
            for key in central_node.model.state_dict().keys():
                if 'bn' not in key:
                    if 'num_batches_tracked' in key:
                        central_node.model.state_dict()[key].data.copy_(client_nodes[0].model.state_dict()[key])
                    else:
                        temp = torch.zeros_like(central_node.model.state_dict()[key], dtype=torch.float32)
                        for i, client_idx in enumerate(select_list):
                            temp += agg_weights[i] * client_nodes[client_idx].model.state_dict()[key]
                        central_node.model.state_dict()[key].data.copy_(temp)
                        for i, client_idx in enumerate(select_list):
                            client_nodes[client_idx].model.state_dict()[key].data.copy_(central_node.model.state_dict()[key])
        elif args.local_model == 'UNet3D':
            for key in central_node.model.state_dict().keys():
                if 'down_1.0.1' not in key and 'down_1.2' not in key and 'down_2.0.1' not in key and 'down_2.2' not in key and 'down_3.0.1' not in key and 'down_3.2' not in key and 'down_4.0.1' not in key and 'down_4.2' not in key and 'bridge.0.1' not in key and 'bridge.2' not in key and 'trans_2.1' not in key and 'up_2.0.1' not in key and 'up_2.2' not in key and 'trans_3.1' not in key and 'up_3.0.1' not in key and 'up_3.2' not in key and 'trans_4.1' not in key and 'up_4.0.1' not in key and 'up_4.2' not in key and 'trans_5.1' not in key and 'up_5.0.1' not in key and 'up_5.2' not in key:
                    if 'num_batches_tracked' in key:
                        central_node.model.state_dict()[key].data.copy_(client_nodes[0].model.state_dict()[key])
                    else:
                        temp = torch.zeros_like(central_node.model.state_dict()[key], dtype=torch.float32)
                        for i, client_idx in enumerate(select_list):
                            temp += agg_weights[i] * client_nodes[client_idx].model.state_dict()[key]
                        central_node.model.state_dict()[key].data.copy_(temp)
                        for i, client_idx in enumerate(select_list):
                            client_nodes[client_idx].model.state_dict()[key].data.copy_(central_node.model.state_dict()[key])
        else:
            for key in central_node.model.state_dict().keys():
                if 'num_batches_tracked' in key:
                    central_node.model.state_dict()[key].data.copy_(client_nodes[0].model.state_dict()[key])
                else:
                    if 'enc1.1' not in key and 'enc2.1' not in key and 'enc3.1' not in key and 'enc4.1' not in key and 'dec4.1' not in key and 'dec3.1' not in key and 'dec2.1' not in key: # 'enc1.1'
                        temp = torch.zeros_like(central_node.model.state_dict()[key], dtype=torch.float32)
                        for i, client_idx in enumerate(select_list):
                            temp += agg_weights[i] * client_nodes[client_idx].model.state_dict()[key]
                        central_node.model.state_dict()[key].data.copy_(temp)
                        for i, client_idx in enumerate(select_list):
                            client_nodes[client_idx].model.state_dict()[key].data.copy_(central_node.model.state_dict()[key])

    elif args.server_method == 'siobn':
        if args.local_model == 'UNet2D' or args.local_model == 'SANet':
            for key in central_node.model.state_dict().keys():
                if 'bn' not in key:
                    if 'num_batches_tracked' in key:
                        central_node.model.state_dict()[key].data.copy_(client_nodes[0].model.state_dict()[key])
                    else:
                        temp = torch.zeros_like(central_node.model.state_dict()[key], dtype=torch.float32)
                        for i, client_idx in enumerate(select_list):
                            temp += agg_weights[i] * client_nodes[client_idx].model.state_dict()[key]
                        central_node.model.state_dict()[key].data.copy_(temp)
                        for i, client_idx in enumerate(select_list):
                            client_nodes[client_idx].model.state_dict()[key].data.copy_(central_node.model.state_dict()[key])
                else:
                    if 'weight' in key or 'bias' in key: # ignore  running_mean, running_var
                        temp = torch.zeros_like(central_node.model.state_dict()[key], dtype=torch.float32)
                        for i, client_idx in enumerate(select_list):
                            temp += agg_weights[i] * client_nodes[client_idx].model.state_dict()[key]
                        central_node.model.state_dict()[key].data.copy_(temp)
                        for i, client_idx in enumerate(select_list):
                            client_nodes[client_idx].model.state_dict()[key].data.copy_(central_node.model.state_dict()[key])
        elif args.local_model == 'UNet3D':
            for key in central_node.model.state_dict().keys():
                if 'down_1.0.1' not in key and 'down_1.2' not in key and 'down_2.0.1' not in key and 'down_2.2' not in key and 'down_3.0.1' not in key and 'down_3.2' not in key and 'down_4.0.1' not in key and 'down_4.2' not in key and 'bridge.0.1' not in key and 'bridge.2' not in key and 'trans_2.1' not in key and 'up_2.0.1' not in key and 'up_2.2' not in key and 'trans_3.1' not in key and 'up_3.0.1' not in key and 'up_3.2' not in key and 'trans_4.1' not in key and 'up_4.0.1' not in key and 'up_4.2' not in key and 'trans_5.1' not in key and 'up_5.0.1' not in key and 'up_5.2' not in key:
                    if 'num_batches_tracked' in key:
                        central_node.model.state_dict()[key].data.copy_(client_nodes[0].model.state_dict()[key])
                    else:
                        temp = torch.zeros_like(central_node.model.state_dict()[key], dtype=torch.float32)
                        for i, client_idx in enumerate(select_list):
                            temp += agg_weights[i] * client_nodes[client_idx].model.state_dict()[key]
                        central_node.model.state_dict()[key].data.copy_(temp)
                        for i, client_idx in enumerate(select_list):
                            client_nodes[client_idx].model.state_dict()[key].data.copy_(central_node.model.state_dict()[key])
                else:
                    if 'weight' in key or 'bias' in key: # ignore  running_mean, running_var
                        temp = torch.zeros_like(central_node.model.state_dict()[key], dtype=torch.float32)
                        for i, client_idx in enumerate(select_list):
                            temp += agg_weights[i] * client_nodes[client_idx].model.state_dict()[key]
                        central_node.model.state_dict()[key].data.copy_(temp)
                        for i, client_idx in enumerate(select_list):
                            client_nodes[client_idx].model.state_dict()[key].data.copy_(central_node.model.state_dict()[key])
        else:
            for key in central_node.model.state_dict().keys():
                if 'enc1.1' not in key and 'enc2.1' not in key and 'enc3.1' not in key and 'enc4.1' not in key and 'dec4.1' not in key and 'dec3.1' not in key and 'dec2.1' not in key: # 'enc1.1'
                    if 'num_batches_tracked' in key:
                        central_node.model.state_dict()[key].data.copy_(client_nodes[0].model.state_dict()[key])
                    else:
                        temp = torch.zeros_like(central_node.model.state_dict()[key], dtype=torch.float32)
                        for i, client_idx in enumerate(select_list):
                            temp += agg_weights[i] * client_nodes[client_idx].model.state_dict()[key]
                        central_node.model.state_dict()[key].data.copy_(temp)
                        for i, client_idx in enumerate(select_list):
                            client_nodes[client_idx].model.state_dict()[key].data.copy_(central_node.model.state_dict()[key])
                else:
                    if 'weight' in key or 'bias' in key: # ignore  running_mean, running_var
                        temp = torch.zeros_like(central_node.model.state_dict()[key], dtype=torch.float32)
                        for i, client_idx in enumerate(select_list):
                            temp += agg_weights[i] * client_nodes[client_idx].model.state_dict()[key]
                        central_node.model.state_dict()[key].data.copy_(temp)
                        for i, client_idx in enumerate(select_list):
                            client_nodes[client_idx].model.state_dict()[key].data.copy_(central_node.model.state_dict()[key])

    elif args.server_method == 'singleset':
        pass
    else:
        raise ValueError('Undefined server method...')
        
    return central_node, client_nodes
