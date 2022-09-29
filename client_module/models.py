import math
import torch
import torch.nn.functional as F
import torch.nn as nn

def create_model_instance(dataset_type, model_type, class_num=10):
    # return VGG9()
    # return EMNIST_CNN1(),EMNIST_CNN2()
    return AlexNet_DF1(),AlexNet_DF2()
    # return IMAGE100_VGG16_1(),IMAGE100_VGG16_2()

class AlexNet_DF1(nn.Module):
    def __init__(self):
        super(AlexNet_DF1, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        x = self.features(x)
        return x
    
class AlexNet_DF2(nn.Module):
    def __init__(self, class_num=10):
        super(AlexNet_DF2, self).__init__()
        
        # self.f2= nn.Sequential(
            
        # )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, class_num),
        )

    def forward(self, x):
        # x = self.f2(x)
        x = x.view(x.size(0), 256 * 4 * 4)
        x = self.classifier(x)
        return x

class EMNIST_CNN1(nn.Module):
    def __init__(self):
        super(EMNIST_CNN1,self).__init__()

        self.conv1 = nn.Sequential(        
            nn.Conv2d(1,32,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.conv2 = nn.Sequential(        
            nn.Conv2d(32,64,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

    def forward(self,x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        return out_conv2

class EMNIST_CNN2(nn.Module):
    def __init__(self):
        super(EMNIST_CNN2,self).__init__()
        self.fc1 = nn.Linear(7*7*64,512)
        self.fc2 = nn.Linear(512, 62)

    def forward(self,out_conv2):
        output = out_conv2.view(-1,7*7*64)
        output = F.relu(self.fc1(output))
        output = self.fc2(output)
        return output

class IMAGE100_VGG16_1(nn.Module):
    def __init__(self, class_num=100):
        super(IMAGE100_VGG16_1, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(128, 256, kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(256, 512, kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2,padding=1),
            
            nn.Conv2d(512, 512, kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
        )
    def forward(self, x):
        x = self.features(x)
        return x

class IMAGE100_VGG16_2(nn.Module):
    def __init__(self, class_num=100):
        super(IMAGE100_VGG16_2, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(512*25, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 100)
        )

    def forward(self, x):
        x = x.view(x.size(0), 512*25)
        x = self.classifier(x)
        return x
