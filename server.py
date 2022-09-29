import os
import sys
import argparse
import socket
import pickle
import asyncio
import concurrent.futures
import json
import random
import time
import numpy as np
import threading
import torch
import copy
import math
from config import *
import torch.optim as optim
import torch.nn.functional as F
import datasets, models
from training_utils import test2
import torch.nn as nn

#init parameters
parser = argparse.ArgumentParser(description='Distributed Client')
parser.add_argument('--dataset_type', type=str, default='CIFAR10')
parser.add_argument('--model_type', type=str, default='AlexNet')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--data_pattern', type=int, default=4)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--decay_rate', type=float, default=0.993)
parser.add_argument('--min_lr', type=float, default=0.005)
parser.add_argument('--epoch', type=int, default=250)
parser.add_argument('--momentum', type=float, default=-1)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--use_cuda', action="store_false", default=True)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
master_listen_port_base=53710
RESULT_PATH = 'result_record'

def main():

    # init config
    common_config = CommonConfig()
    common_config.model_type = args.model_type
    common_config.dataset_type = args.dataset_type
    common_config.batch_size = args.batch_size
    common_config.data_pattern=args.data_pattern
    common_config.lr = args.lr
    common_config.decay_rate = args.decay_rate
    common_config.min_lr=args.min_lr
    common_config.epoch = args.epoch
    common_config.momentum = args.momentum
    common_config.weight_decay = args.weight_decay

    #read the worker_config.json to init the worker node
    with open("worker_config.json") as json_file:
        workers_config = json.load(json_file)

    worker_num = len(workers_config['worker_config_list'])

    client_model,global_model = models.create_model_instance(common_config.dataset_type, common_config.model_type)
    init_para = torch.nn.utils.parameters_to_vector(client_model.parameters())
    global_model.to(device)

    common_config.para_nums=init_para.nelement()
    model_size = init_para.nelement() * 4 / 1024 / 1024
    print("para num: {}".format(common_config.para_nums))
    print("Model Size: {} MB".format(model_size))

    init_para1 = torch.nn.utils.parameters_to_vector(global_model.parameters())
    model_size = init_para1.nelement() * 4 / 1024 / 1024
    print("para num: {}".format(common_config.para_nums))
    print("Model Size: {} MB".format(model_size))
    # create workers
    worker_list: List[Worker] = list()
    for worker_idx, worker_config in enumerate(workers_config['worker_config_list']):
        custom = dict()
        custom["computation"] = worker_config["computation"]
        custom["dynamics"] = worker_config["dynamics"]
        worker_list.append(
            Worker(config=ClientConfig(common_config=common_config,custom=custom),
                    idx=worker_idx,
                    client_ip=worker_config['ip_address'],
                    user_name=worker_config['user_name'],
                    pass_wd=worker_config['pass_wd'],
                    remote_scripts_path=workers_config['scripts_path']['remote'],
                    master_port=master_listen_port_base+worker_idx,
                    location='local'
                    )
        )
    #到了这里，worker已经启动了

    # Create model instance
    train_data_partition = partition_data(common_config.dataset_type, common_config.data_pattern)

    for worker_idx, worker in enumerate(worker_list):
        worker.config.para = init_para
        worker.config.custom["train_data_idxes"] = train_data_partition.use(worker_idx)
        # worker.config.custom["test_data_idxes"] = test_data_partition.use(worker_idx)

    # connect socket and send init config
    communication_parallel(worker_list, action="init")

    recoder: SummaryWriter = SummaryWriter()
    global_model.to(device)
    _, test_dataset = datasets.load_datasets(common_config.dataset_type)
    test_loader = datasets.create_dataloaders(test_dataset, batch_size=128, shuffle=False)

    total_time=0.0
    local_steps_list=[50,50,50,50,50,50,50,50,50,50]
    compre_ratio_list=[1,1,1,1,1,1,1,1,1,1]
    computation_resource=[3,1,6,7,7,5,5,2,6,2]
    bandwith_resource=[5,6,8,1,5,8,2,4,4,2]
    total_resource=0.0
    total_bandwith=0.0
    #computation_resource,bandwith_resource=random_RC(10)

    path=os.getcwd()
    print (path)
    path=path+"//"+RESULT_PATH
    if not os.path.exists(path):
        os.makedirs(path)
    now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
    path=path+"//"+now+"_record.txt"
    result_out = open(path, 'a+')

    #result_out.write(common_config.__dict__)
    print(common_config.__dict__,file=result_out)
    result_out.write('\n')
    result_out.write("epoch_idx, total_time, total_bandwith, total_resource, acc, test_loss")
    result_out.write('\n')
    local_steps=20

    epoch_lr = args.lr
    for epoch_idx in range(1, 1+common_config.epoch):

        communication_parallel(worker_list, action="send_para", data=local_steps)

        if epoch_idx > 1 and epoch_idx % 1 == 0:
            epoch_lr = max((args.decay_rate * args.lr, args.min_lr))
        optimizer = optim.SGD(global_model.parameters(), lr=epoch_lr, weight_decay=args.weight_decay)
        total_resource,total_bandwith,total_time=train2(global_model, device, worker_list,epoch_lr, local_steps,total_resource,total_bandwith,total_time)

        print("get begin")
        communication_parallel(worker_list, action="get_para")
        communication_parallel(worker_list, action="get_time")
        print("get end")
        
        
        para_delta = aggregate_model_para2(client_model, worker_list, device)
        # global_para = aggregate_compressed_model(global_model,worker_list)
        print("send begin")
        communication_parallel(worker_list, action="send_model",data=para_delta)
        print("send end")
        
        test_loss, acc = test2(global_model,client_model, test_loader, device)
        recoder.add_scalar('Accuracy/average', acc, epoch_idx)
        recoder.add_scalar('Test_loss/average', test_loss, epoch_idx)
        print("Epoch: {}, accuracy: {}, test_loss: {}\n".format(epoch_idx, acc, test_loss))

        local_steps,sum_time=update_E(worker_list)
        total_time=total_time+sum_time
        total_resource=total_resource+Sum(computation_resource,local_steps_list)
        total_bandwith=total_bandwith+Sum(bandwith_resource,compre_ratio_list)
        print("total_time: {}, total_resource: {}, total_bandwith: {}\n".format(total_time,total_resource,total_bandwith))
        recoder.add_scalar('Accuracy/average_time', acc, total_time)
        recoder.add_scalar('Test_loss/average_time', test_loss, total_time)
        recoder.add_scalar('resource_time', total_resource, total_time)
        recoder.add_scalar('bandwith_time', total_bandwith, total_time)
        recoder.add_scalar('resource_epoch', total_resource, epoch_idx)
        recoder.add_scalar('bandwith_epoch', total_bandwith, epoch_idx)
        result_out.write('{} {:.2f} {:.2f} {:.2f} {:.4f} {:.4f}'.format(epoch_idx,total_time,total_bandwith,total_resource,acc,test_loss))
        result_out.write('\n')

        print(local_steps_list)
        print(compre_ratio_list)
        
    # close socket
    result_out.close()
    for worker in worker_list:
        worker.socket.shutdown(2)
    
def Sum(list1,list2):
    sum=0.0
    for idx in range(0, len(list1)):
        sum=sum+float(list1[idx])*float(list2[idx])
    return sum

def random_RC(num):
    computation_resource=np.random.randint(1,num,num)
    bandwith_resourc=np.random.randint(1,num,num)
    return computation_resource,bandwith_resourc

def update_E(worker_list):
    local_steps=random.randint(40,60)
    compre_ratio=local_steps/200.0
    train_time_list=[0.8,0.7,0.8,0.6,0.8,0.7,0.6,0.8,0.7,0.6]
    send_time_list=[0.8,0.7,0.8,0.6,0.8,0.7,0.6,0.8,0.7,0.6]
    min_train_time=10000.0
    min_train_time_idx=1
    min_send_time=10000.0
    min_send_time_idx=1
    sum_local_steps=0
    for worker in worker_list:
        train_time_list[worker.idx]=worker.config.train_time
        send_time_list[worker.idx]=worker.config.send_time
        if train_time_list[worker.idx]<min_train_time:
            min_train_time=train_time_list[worker.idx]
            min_train_time_idx=worker.idx
        if send_time_list[worker.idx]<min_send_time:
            min_send_time=send_time_list[worker.idx]
            min_send_time_idx=worker.idx
    for worker in worker_list:
        worker.config.batch_size=int((train_time_list[min_train_time_idx]/train_time_list[worker.idx])*local_steps)
        worker.config.compre_ratio=(train_time_list[min_train_time_idx]/train_time_list[worker.idx])*compre_ratio
        #(send_time_list[min_train_time_idx]/send_time_list[worker.idx])*compre_ratio
        # worker.config.batch_size=5
        #worker.config.batch_size=int(local_steps/2)+3
        # worker.config.compre_ratio=1.0
        # local_steps_list[worker.idx]=worker.config.batch_size
        # compre_ratio_list[worker.idx]=worker.config.compre_ratio
        sum_local_steps=sum_local_steps+worker.config.batch_size
    for worker in worker_list:
        worker.config.average_weight=(1.0*worker.config.batch_size)/(sum_local_steps)
        
    max_train_time=max(train_time_list)
    max_send_time=max(send_time_list)
    total_time=max_train_time*4
    #local_steps/2*0.9
    #total_time=min_train_time*50+min_train_time*40
    #total_time=min_train_time*local_steps/2.0
    return local_steps,total_time

def update_B(worker_list,batch_size_list,compre_ratio_list):
    batch=128
    compre_ratio=batch/200.0
    train_time_list=[0.8,0.7,0.8,0.6,0.8,0.7,0.6,0.8,0.7,0.6]
    send_time_list=[0.8,0.7,0.8,0.6,0.8,0.7,0.6,0.8,0.7,0.6]
    min_train_time=10000.0
    min_train_time_idx=1
    min_send_time=10000.0
    min_send_time_idx=1
    sum_local_steps=0
    for worker in worker_list:
        train_time_list[worker.idx]=worker.config.train_time
        send_time_list[worker.idx]=worker.config.send_time
        if train_time_list[worker.idx]<min_train_time:
            min_train_time=train_time_list[worker.idx]
            min_train_time_idx=worker.idx
        if send_time_list[worker.idx]<min_send_time:
            min_send_time=send_time_list[worker.idx]
            min_send_time_idx=worker.idx
    for worker in worker_list:
        worker.config.batch_size=int((train_time_list[min_train_time_idx]/train_time_list[worker.idx])*batch)
        worker.config.compre_ratio=(train_time_list[min_train_time_idx]/train_time_list[worker.idx])*compre_ratio
        #(send_time_list[min_train_time_idx]/send_time_list[worker.idx])*compre_ratio
        # worker.config.batch_size=5
        #worker.config.batch_size=int(local_steps/2)+3
        # worker.config.compre_ratio=1.0
        batch_size_list[worker.idx]=worker.config.batch_size
        compre_ratio_list[worker.idx]=worker.config.compre_ratio
        sum_local_steps=sum_local_steps+worker.config.batch_size
    for worker in worker_list:
        worker.config.average_weight=(1.0*worker.config.batch_size)/(sum_local_steps)
        
    max_train_time=max(train_time_list)
    max_send_time=max(send_time_list)
    total_time=max_train_time*4
    #local_steps/2*0.9
    #total_time=min_train_time*50+min_train_time*40
    #total_time=min_train_time*local_steps/2.0
    return local_steps,total_time

def aggregate_model_para(global_model, worker_list):
    global_para = torch.nn.utils.parameters_to_vector(global_model.parameters()).detach()
    with torch.no_grad():
        para_delta = torch.zeros_like(global_para)
        for worker in worker_list:
            model_delta = (worker.config.neighbor_paras - global_para)
            para_delta += worker.config.average_weight * model_delta
            #print(para_delta)
        global_para += para_delta
    torch.nn.utils.vector_to_parameters(global_para, global_model.parameters())
    return global_para

def aggregate_model_dict(global_model,worker_list):
    with torch.no_grad():
        local_model_para = []
        for worker in worker_list:
            local_model_para.append(worker.config.neighbor_paras)
        para_delta = copy.deepcopy(local_model_para[0])
        for para in para_delta.keys():
            para_delta[para] = para_delta[para]*0.0
            for p in local_model_para:
                para_delta[para] = para_delta[para]*1.0 + p[para]*1.0
            para_delta[para] = para_delta[para] / (len(local_model_para))
    global_model.load_state_dict(para_delta)
    return para_delta

def aggregate_compressed_model(global_model, worker_list):
    global_para = torch.nn.utils.parameters_to_vector(global_model.parameters()).detach()
    with torch.no_grad():
        para_delta = torch.zeros_like(global_para)
        for worker in worker_list:
            indice = worker.config.neighbor_indices
            selected_indicator = torch.zeros_like(global_para)
            selected_indicator[indice] = 1.0
            # model_delta = (worker.config.neighbor_paras - global_para) * selected_indicator
            #gradient
            model_delta = worker.config.neighbor_paras
            para_delta += worker.config.average_weight * model_delta
        global_para += para_delta
    torch.nn.utils.vector_to_parameters(global_para, global_model.parameters())
    return global_para

def communication_parallel(worker_list, action, data=None):
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(worker_list),)
        tasks = []
        for worker in worker_list:
            if action == "init":
                tasks.append(loop.run_in_executor(executor, worker.send_init_config))
            elif action == "get_para":
                tasks.append(loop.run_in_executor(executor, get_model,worker))
            elif action == "get_time":
                tasks.append(loop.run_in_executor(executor, get_time,worker))
            elif action == "get_data_feature":
                tasks.append(loop.run_in_executor(executor, get_data_feature,worker))
            elif action == "send_model":
                tasks.append(loop.run_in_executor(executor, worker.send_data, data))
            elif action == "send_para":
                data=worker.config.batch_size
                tasks.append(loop.run_in_executor(executor, worker.send_data,data))
        loop.run_until_complete(asyncio.wait(tasks))
        loop.close()
    except:
        sys.exit(0)

def get_time(worker):
    train_time,send_time= get_data_socket(worker.socket)
    worker.config.train_time=train_time
    worker.config.send_time=send_time
    print(worker.idx," train time: ", train_time," send time: ", send_time)

def get_compressed_model_top(worker):
    nelement=worker.config.common_config.para_nums
    received_para, indices = get_data_socket(worker.socket)
    received_para.to(device)

    restored_model = torch.zeros(nelement).to(device)
    
    restored_model[indices] = received_para

    worker.config.neighbor_paras = restored_model.data
    worker.config.neighbor_indices = indices

def get_model(worker):
    received_para = get_data_socket(worker.socket)
    worker.config.neighbor_paras = received_para.to(device)
    # print(worker.config.neighbor_paras)

def non_iid_partition111(ratio, worker_num=10):
    partition_sizes = np.ones((10, worker_num)) * ((1 - ratio) / (worker_num-1))

    for worker_idx in range(worker_num):
        partition_sizes[worker_idx][worker_idx] = ratio

    return partition_sizes

def non_iid_partition(ratio, train_class_num, worker_num):
    partition_sizes = np.ones((train_class_num, worker_num)) * ((1 - ratio) / (worker_num-1))

    for i in range(train_class_num):
        partition_sizes[i][i%worker_num]=ratio

    return partition_sizes

def partition_data(dataset_type, data_pattern, worker_num=10):
    train_dataset, _ = datasets.load_datasets(dataset_type)

    if dataset_type == "CIFAR10" or dataset_type == "FashionMNIST":
        train_class_num=10
        if data_pattern == 0:
            partition_sizes = np.ones((train_class_num, worker_num)) * (1.0 / worker_num)
        elif data_pattern == 1:
            non_iid_ratio = 0.2
            partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
        elif data_pattern == 2:
            non_iid_ratio = 0.4
            partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
        elif data_pattern == 3:
            non_iid_ratio = 0.6
            partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
        elif data_pattern == 4:
            non_iid_ratio = 0.8
            partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
    elif dataset_type == "EMNIST":
        train_class_num=62
        if data_pattern == 0:
            partition_sizes = np.ones((train_class_num, worker_num)) * (1.0 / worker_num)
        elif data_pattern == 1:
            non_iid_ratio = 0.2
            partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
        elif data_pattern == 2:
            non_iid_ratio = 0.4
            partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
        elif data_pattern == 3:
            non_iid_ratio = 0.6
            partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
        elif data_pattern == 4:
            non_iid_ratio = 0.8
            partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
    if dataset_type == "CIFAR100" or dataset_type == "image100":
        train_class_num=100
        if data_pattern == 0:
            partition_sizes = np.ones((train_class_num, worker_num)) * (1.0 / worker_num)
        elif data_pattern == 1:
            non_iid_ratio = 0.2
            partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
        elif data_pattern == 2:
            non_iid_ratio = 0.4
            partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
        elif data_pattern == 3:
            non_iid_ratio = 0.6
            partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
        elif data_pattern == 4:
            non_iid_ratio = 0.8
            partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
    train_data_partition = datasets.LabelwisePartitioner(train_dataset, partition_sizes=partition_sizes)
    return train_data_partition

def partition_data11(dataset_type, data_pattern, worker_num=10):
    train_dataset, test_dataset = datasets.load_datasets(dataset_type)

    if dataset_type == "CIFAR100" or dataset_type == "image100":
        test_partition_sizes = np.ones((100, worker_num)) * (1 / worker_num)
        partition_sizes = np.ones((100, worker_num)) * (1 / (worker_num-data_pattern))
        for worker_idx in range(worker_num):
            tmp_idx = worker_idx
            for _ in range(data_pattern):
                partition_sizes[tmp_idx*worker_num:(tmp_idx+1)*worker_num, worker_idx] = 0
                tmp_idx = (tmp_idx + 1) % 10
                
    elif dataset_type == 'tinyImageNet':
        test_partition_sizes = np.ones((200, worker_num)) * (1 / worker_num)
        partition_sizes = np.ones((200, worker_num))
        for worker_idx in range(worker_num):
            tmp_idx = worker_idx*20
            for _ in range(int(data_pattern/10)):
                partition_sizes[tmp_idx:tmp_idx+10, worker_idx] = 0
                tmp_idx = (tmp_idx + 10) % 200
        axis = np.sum(partition_sizes, axis=1)
        for i in range(200):
            for j in range(worker_num):
                if partition_sizes[i][j] == 1:
                    partition_sizes[i][j] = 1/axis[i]

    elif dataset_type == "EMNIST":
        test_partition_sizes = np.ones((62, worker_num)) * (1 / worker_num)
        partition_sizes = np.ones((62, worker_num))
        for worker_idx in range(worker_num):
            tmp_idx = worker_idx*6
            for _ in range(int(data_pattern/2)):
                partition_sizes[tmp_idx:tmp_idx+2, worker_idx] = 0
                tmp_idx = (tmp_idx + 2) % 62
        axis = np.sum(partition_sizes, axis=1)
        for i in range(62):
            for j in range(worker_num):
                if partition_sizes[i][j] == 1:
                    partition_sizes[i][j] = 1/axis[i]

    elif dataset_type == "CIFAR10" or dataset_type == "FashionMNIST":
        test_partition_sizes = np.ones((10, worker_num)) * (1 / worker_num)
        if data_pattern == 0:
            partition_sizes = np.ones((10, worker_num)) * (1.0 / worker_num)
        elif data_pattern == 1:
            partition_sizes = [
                                [0.0,    0.0,    0.0,    0.1482, 0.1482, 0.1482, 0.148, 0.1482, 0.1482,0.111],
                                [0.0,    0.0,    0.0,    0.1482, 0.1482, 0.1482, 0.1482, 0.148, 0.1482,0.111],
                                [0.0,    0.0,    0.0,    0.1482, 0.1482, 0.1482, 0.1482, 0.1482, 0.148,0.111],
                                [0.148, 0.1482, 0.1482, 0.0,    0.0,    0.0,    0.1482, 0.1482, 0.1482,0.111],
                                [0.1482, 0.148, 0.1482, 0.0,    0.0,    0.0,    0.1482, 0.1482, 0.1482,0.111],
                                [0.1482, 0.1482, 0.148, 0.0,    0.0,    0.0,    0.1482, 0.1482, 0.1472,0.112],
                                [0.1482,  0.1482, 0.1482, 0.148, 0.1482, 0.1482, 0.0,    0.0,    0.0  , 0.111],
                                [0.1482,  0.1482, 0.1482, 0.1482, 0.148, 0.1482, 0.0,    0.0,    0.0  , 0.111],
                                [0.1482,  0.1482, 0.1482, 0.1482, 0.1482, 0.148, 0.0,    0.0,    0.0  , 0.111],
                                [0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.112, 0.0],
                                ]
        elif data_pattern == 2:
            partition_sizes = [
                    [0.0,   0.0,   0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
                    [0.0,   0.0,   0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
                    [0.125, 0.125, 0.0,   0.0,   0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
                    [0.125, 0.125, 0.0,   0.0,   0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
                    [0.125, 0.125, 0.125, 0.125, 0.0,   0.0,   0.125, 0.125, 0.125, 0.125],
                    [0.125, 0.125, 0.125, 0.125, 0.0,   0.0,   0.125, 0.125, 0.125, 0.125],
                    [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.0,   0.0,   0.125, 0.125],
                    [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.0,   0.0,   0.125, 0.125],
                    [0.125,  0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.0,   0.0],
                    [0.125,  0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.0,   0.0],
                    ]
        elif data_pattern == 3:
            partition_sizes = [[0.1428,  0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1432, 0.0,    0.0,    0.0],
                                [0.0,    0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1432, 0.0,    0.0],
                                [0.0,    0.0,    0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1432, 0.0],
                                [0.0,    0.0,    0.0,    0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1432],
                                [0.1432, 0.0,    0.0,    0.0,    0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1428],
                                [0.1428, 0.1432, 0.0,    0.0,    0.0,    0.1428, 0.1428, 0.1428, 0.1428, 0.1428],
                                [0.1428, 0.1428, 0.1432, 0.0,    0.0,    0.0,    0.1428, 0.1428, 0.1428, 0.1428],
                                [0.1428, 0.1428, 0.1428, 0.1432, 0.0,    0.0,    0.0,    0.1428, 0.1428, 0.1428],
                                [0.1428, 0.1428, 0.1428, 0.1428, 0.1432, 0.0,    0.0,    0.0,    0.1428, 0.1428],
                                [0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1432, 0.0,    0.0,    0.0,    0.1428],
                                ]
        elif data_pattern == 4:
            partition_sizes = [[0.125,  0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.0,   0.0],
                                [0.0,   0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.0],
                                [0.0,   0.0,   0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
                                [0.125, 0.0,   0.0,   0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
                                [0.125, 0.125, 0.0,   0.0,   0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
                                [0.125, 0.125, 0.125, 0.0,   0.0,   0.125, 0.125, 0.125, 0.125, 0.125],
                                [0.125, 0.125, 0.125, 0.125, 0.0,   0.0,   0.125, 0.125, 0.125, 0.125],
                                [0.125, 0.125, 0.125, 0.125, 0.125, 0.0,   0.0,   0.125, 0.125, 0.125],
                                [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.0,   0.0,   0.125, 0.125],
                                [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.0,   0.0,   0.125],
                                ]
        elif data_pattern == 5:
            non_iid_ratio = 0.2
            partition_sizes = non_iid_partition(non_iid_ratio)
        elif data_pattern == 6:
            non_iid_ratio = 0.4
            partition_sizes = non_iid_partition(non_iid_ratio)
        elif data_pattern == 7:
            non_iid_ratio = 0.6
            partition_sizes = non_iid_partition(non_iid_ratio)
        elif data_pattern == 8:
            non_iid_ratio = 0.8
            partition_sizes = non_iid_partition(non_iid_ratio)
        elif data_pattern == 9:
            non_iid_ratio = 0.9
            partition_sizes = non_iid_partition(non_iid_ratio)
        # elif data_pattern == 10:
        #     non_iid_ratio = 0.5
        #     partition_sizes = non_iid_partition(non_iid_ratio)

    train_data_partition = datasets.LabelwisePartitioner(train_dataset, partition_sizes=partition_sizes)
    # test_data_partition = datasets.LabelwisePartitioner(test_dataset, partition_sizes=partition_sizes)
    test_data_partition = datasets.LabelwisePartitioner(test_dataset, partition_sizes=test_partition_sizes)
    
    return train_data_partition, test_data_partition

def get_data_feature(worker):
    received_para = get_data_socket(worker.socket)
    worker.config.neighbor_paras = received_para

def train2(model, device, worker_list,epoch_lr,local_steps):
    # model1,model2 = models.create_model_instance()

    batch_size_list=[50,50,50,50,50,50,50,50,50,50]
    compre_ratio_list=[1,1,1,1,1,1,1,1,1,1]
    computation_resource=[3,1,6,7,7,5,5,2,6,2]
    bandwith_resource=[5,6,8,1,5,8,2,4,4,2]
    train_loss = 0.0
    samples_num = 0
    loss_func = nn.CrossEntropyLoss() 
    for iter_idx in range(local_steps):
        communication_parallel(worker_list, action="send_para")
        origin_para = torch.nn.utils.parameters_to_vector(model.parameters()).detach()      # 保存原来的参数
        communication_parallel(worker_list, action="get_data_feature")
        paras = []
        for worker in worker_list:
            # print(worker.config.neighbor_paras)
            model1,model2 = models.create_model_instance("a","b")
            torch.nn.utils.vector_to_parameters(origin_para,model2.parameters())
            optimizer = optim.SGD(model2.parameters(), lr=epoch_lr, weight_decay=args.weight_decay)
            data_feature,target = worker.config.neighbor_paras[0].to(device),worker.config.neighbor_paras[1].to(device)
            input = data_feature.detach().requires_grad_()          # 将输入转换成有梯度的形式

            optimizer.zero_grad()
            output1 = model2(input)
            loss =loss_func(output1, target)
            loss.backward()                 
        
            grad_in = input.grad               
            optimizer.step()
            worker.send_data(grad_in)           
            paras.append(copy.deepcopy(model2))
        
        batch_size_list,sum_time=update_B(worker_list,batch_size_list,compre_ratio_list)
        total_time=total_time+sum_time
        total_resource=total_resource+Sum(computation_resource,batch_size_list)
        total_bandwith=total_bandwith+Sum(bandwith_resource,batch_size_list)

        vector = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
        vector = vector * 0.0
        for para in paras:
            new_para = torch.nn.utils.parameters_to_vector(para.parameters()).detach()
            vector += new_para*0.1
        torch.nn.utils.vector_to_parameters(vector,model.parameters())

    print("forward and back prpagation")
    if samples_num != 0:
        train_loss /= samples_num
    
    # return train_loss
    return total_resource,total_bandwith,total_time

def aggregate_model_para2(client_model, worker_list,device):
    global_para = torch.nn.utils.parameters_to_vector(client_model.parameters()).detach()
    with torch.no_grad():
        para_delta = torch.zeros_like(global_para).to(device)
        for worker in worker_list:
            para_delta += 0.1 * worker.config.neighbor_paras
    torch.nn.utils.vector_to_parameters(para_delta, client_model.parameters())
    return para_delta


if __name__ == "__main__":

    # global_model = models.create_model_instance('CIFAR10', 'AlexNet')
    # print(global_model.state_dict())
    main()
