import sys
import time
import math
import re
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from client_comm_utils import *



def train(model, data_loader, optimizer, local_iters=None, device=torch.device("cpu"), model_type=None):
    t_start = time.time()
    model.train()
    if local_iters is None:
        local_iters = math.ceil(len(data_loader.loader.dataset) / data_loader.loader.batch_size)
    #print("local_iters: ", local_iters)

    train_loss = 0.0
    samples_num = 0
    for iter_idx in range(local_iters):
        data, target = next(data_loader)

        if model_type == 'LR':
            data = data.squeeze(1).view(-1, 28 * 28)
        
        # target=target%5

        data, target = data.to(device), target.to(device)
        
        output = model(data)

        optimizer.zero_grad()
        loss_func = nn.CrossEntropyLoss() 
        loss =loss_func(output, target)
        #loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        train_loss += (loss.item() * data.size(0))
        samples_num += data.size(0)

    if samples_num != 0:
        train_loss /= samples_num
    
    return train_loss

def test(model, data_loader, device=torch.device("cpu"), model_type=None):
    model.eval()
    data_loader = data_loader.loader
    test_loss = 0.0
    test_accuracy = 0.0

    correct = 0

    with torch.no_grad():
        for data, target in data_loader:

            data, target = data.to(device), target.to(device)


            # target=target%5

            if model_type == 'LR':
                data = data.squeeze(1).view(-1, 28 * 28)
            output = model(data)

            # sum up batch loss
            loss_func = nn.CrossEntropyLoss(reduction='sum') 
            test_loss += loss_func(output, target).item()
            #test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(1, keepdim=True)
            batch_correct = pred.eq(target.view_as(pred)).sum().item()

            correct += batch_correct
            

    test_loss /= len(data_loader.dataset)
    test_accuracy = np.float(1.0 * correct / len(data_loader.dataset))

    # TODO: Record

    return test_loss, test_accuracy

def train2(model, train_data, train_label, optimizer, local_iters, device, master_socket, start_idx,train_loader):
    t_start = time.time()
    model.train()
    train_loss = 0.0
    samples_num = len(train_label)
    
    for iter_idx in range(local_iters):
        batch_size=get_data_socket(master_socket)
        data = torch.reshape(train_data[start_idx:start_idx+batch_size, :, :], [-1, 32, 32]).to(device)
        target = (train_label[start_idx:start_idx+batch_size]).to(device)
        start_idx=start_idx+batch_size
        if start_idx>=samples_num:
            start_idx=0
        
        data,target=next(train_loader)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()


        # data = data.to(device)
        data_feature = model(data)

        # input = output0.detach().requires_grad_()
        # x_data = data_feature.to(torch.device("cpu")) 
        x_data = data_feature.detach().requires_grad_().to(torch.device("cpu")) 
        target = target.to(torch.device("cpu"))
        output = x_data.view(-1).detach()
        print("feature num: ",len(output))
        print("feacture size: ",len(output)*4/1024/1024)
        print("send")
        # print(x_data)
        
        send_data_socket((x_data,target), master_socket)            # 发送输出到server
        grad_in = get_data_socket(master_socket)                    # 接收反向传播
        grad_in.to(device)

        data_feature.backward(grad_in)
        optimizer.step()
    
    # return train_loss

# test_loss, acc = test2(global_model,client_model, test_loader, device)

