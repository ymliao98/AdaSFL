import os
import time
import socket
import pickle
import argparse
import asyncio
import concurrent.futures
import threading
import math
import copy
import numpy as np
import torch
import torch.optim as optim
import datasets, models

train_dataset, test_dataset = datasets.load_datasets('CIFAR10')

train_dataset= torch.utils.data.Subset(train_dataset,np.arange(10))
train_loader = datasets.create_dataloaders(train_dataset, batch_size=1)

for i in range(10):
    data1, target1 = next(train_loader)