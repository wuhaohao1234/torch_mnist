from torch import nn
from torch.nn import functional as F
import torch


class LeNet(nn.Module):
    '''Lenet网络'''

    def __init__(self):  # 构建层级框架
        super(LeNet, self).__init__()             # 使用super() 函数，调用父类(超类)的init方法，防止重复示例化
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()                    # nn.ReLU作为一个层结构，必须添加到nn.Module容器中才能使用,而F.ReLU则作为一个函数调用
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.relu5 = nn.ReLU()

    def forward(self, x):                       # 前向计算图
        x = self.conv1(x)                       # 6@24*24
        x = self.relu1(x)
        x = self.pool1(x)                       # 6@12*12
        x = self.conv2(x)                       # 16@8*8
        x = self.relu2(x)
        x = self.pool2(x)
        #print("shape = ",x.shape,end='\n')     # 16@4*4
        x = x.view(x.shape[0], -1)              # 1@256
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.fc3(x)
        x = self.relu5(x)                       # 1*10【10分类问题】
        # print("shape = ", x.shape, end='\n')
        return x


# 定义AlexNet网络结构
class AlexNet(nn.Module):
    '''Alexnet网络结构'''

    def __init__(self, width_mult=1):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(            # 1@28*28
            nn.Conv2d(1, 32, 3, padding=1),     # 32@28*28    注意：padding=1，计算方式有变（ceil(28/1)）
            nn.MaxPool2d(2),                    # 32@14*14
            nn.ReLU(inplace=True)               # 改变输入数据的值,节省反复申请与释放内存的空间与时间,只是将原来的地址传递,效率更好
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),    # 64@14*14
            nn.MaxPool2d(2),                    # 64@7*7
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1)     # 128@7*7
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1)    # 256@7*7
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),   # 256@7*7
            nn.MaxPool2d(3, stride=2),           # 256@3*3
            nn.ReLU(inplace=True)
        )

        self.fc1 = nn.Linear(2304, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.layer1(x)                      # 32@14*14
        x = self.layer2(x)                      # 64@7*7
        x = self.layer3(x)                      # 128@7*7
        x = self.layer4(x)                      # 256@7*7
        x = self.layer5(x)                      # 256@3*3
        x = x.view(x.shape[0], -1)              # 1*2304
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)                         # 1*10
        return x


'''
   对于卷积神经网络，CNN每一层都会输出一个C x H x W的特征图，C就是通道，同时也代表卷积核的数量，亦为特征的数量，H和W就是原始图片经过压缩后的图的高度和宽度，而空间注意力就是对于所有的通道，在二维平面上，
对H x W尺寸的特征图学习到一个权重矩阵，对应每个像素都会学习到一个权重。而这些权重代表的就是某个空间位置信息的重要程度 ，将该空间注意力矩阵附加在原来的特征图上，增大有用的特征，弱化无用特征，从而起到特征筛选
和增强的效果。
'''

class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)      # global maxpooling
        self.max_pool = nn.AdaptiveMaxPool2d(1)      # global avgpooling

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1,bias=False),         # 这里并没有使用MLP，而是使用了CNN+Relu的方式，目的在于减少模型所需参数，减少开销，提高运行速度
                                   nn.ReLU(),
                                   nn.Conv2d(in_planes // 16, in_planes, 1,bias=False))      # 后面还要与特征图相乘，所以通道数需要还原到256
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))          # avg_out = 256*1*1
        max_out = self.fc(self.max_pool(x))          # max_out = 256*1*1
        out = avg_out + max_out                      # out = 256*1*1
        return self.sigmoid(out)

'''不同于空间注意力，通道域注意力类似于给每个通道上的特征图都施加一个权重，来代表该通道与关键信息的相关度的话，这个权重越大，则表示相关度越高。在神经网络中，越高的维度特征图尺寸越小，
通道数越多，通道就代表了整个图像的特征信息。如此多的通道信息，对于神经网络来说，要甄别筛选有用的通道信息是很难的，这时如果用一个通道注意力告诉该网络哪些是重要的，往往能起到很好的效果，
这时CV领域做通道注意力往往比空间好的一个原因。'''


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)         # avg_out = 1@7*7     【基于通道的全局池化，防止信息丢失】    不要直接最大池化，不然信息丢失太多，所以要先平均池化，拿到一般信息，然后再最大池化，防止过拟合
        max_out = torch.max(x, dim=1, keepdim=True)[0]       # max_out = 1@7*7
        x = torch.cat([avg_out, max_out], dim=1)             # x = 2*7*7
        x = self.conv1(x)                                    # x = 1*7*7
        return self.sigmoid(x)


class AlexNet_CBAM(nn.Module):
    def __init__(self, width_mult=1):
        super(AlexNet_CBAM, self).__init__()
        self.layer1 = nn.Sequential(                # 1*28*28
            nn.Conv2d(1, 32, 3, padding=1),         # 32*28*28
            nn.MaxPool2d(2),                        # 32*14*14
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),        # 64*14*14
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64*7*7
            nn.ReLU(inplace=True)
        )
        self.ca1 = ChannelAttention(64)
        self.sa1 = SpatialAttention()
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1)       # 128*7*7
        )
        self.ca2 = ChannelAttention(128)
        self.sa2 = SpatialAttention()
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),      # 256*7*7
        )
        self.ca3 = ChannelAttention(256)
        self.sa3 = SpatialAttention()
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),      # 256*7*7
            nn.MaxPool2d(kernel_size=3, stride=2),  # 256*3*3
            nn.ReLU(inplace=True)
        )

        self.fc1 = nn.Linear(2304, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.layer1(x)                        # 32*14*14
        x = self.layer2(x)                        # 64*7*7
        # x = self.ca1(x) * x
        # x = self.sa1(x) * x
        x = self.layer3(x)                        # 128*7*7
        # x = self.ca2(x) * x
        # x = self.sa2(x) * x
        x = self.layer4(x)                         # 256*7*7
        x = self.ca3(x) * x                        # 将ChannelAttention和SpatialAttention放置于第四层效果显著    256*7*7
        x = self.sa3(x) * x                        # 256*7*7
        x = self.layer5(x)                         # 256*3*3
        x = x.view(x.shape[0], -1)                 # 1*2304
        x = self.fc1(x)                            # 1*1024
        x = self.fc2(x)                            # 1
        x = self.fc3(x)                            # 1*10
        return x




# CBAM
class NewChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(NewChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                 nn.ReLU(),
                                 nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.fc2 = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                 nn.ReLU(),
                                 nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.conv = nn.Conv2d(2 * in_planes, in_planes, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc1(self.avg_pool(x))                     # avg_out = 256*1*1
        max_out = self.fc2(self.max_pool(x))                     # max_out = 256*1*1
        out = self.conv(torch.cat([avg_out, max_out], dim=1))    # 512*1*1 --> 256*1*1
        return self.sigmoid(out)

class NewSpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(NewSpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(256 + 2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)             # 1*7*7
        max_out,_  = torch.max(x, dim=1, keepdim=True)           # 1*7*7
        x = torch.cat([x, avg_out, max_out], dim=1)              # 2*7*7
        x = self.conv1(x)                                        # 258*7*7 --> 1*7*7
        return self.sigmoid(x)

class New_AlexNet_CBAM(nn.Module):
    def __init__(self, width_mult=1):
        super(New_AlexNet_CBAM, self).__init__()
        self.layer1 = nn.Sequential(                       # 输入1*28*28
            nn.Conv2d(1, 32, kernel_size=3, padding=1),    # 32*28*28
            nn.MaxPool2d(kernel_size=2, stride=2),         # 32*14*14
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),   # 64*14*14
            nn.MaxPool2d(kernel_size=2, stride=2),         # 64*7*7
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 128*7*7
            nn.ReLU(inplace=True),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 256*7*7
            nn.ReLU(inplace=True),
        )
        self.ca3 = NewChannelAttention(256)
        self.sa3 = NewSpatialAttention()
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 256*7*7
            nn.MaxPool2d(kernel_size=3, stride=2),          # 256*3*3
            nn.ReLU(inplace=True)
        )

        self.fc1 = nn.Linear(256*3*3, 1024)
        self.drop1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(1024, 512)
        self.drop2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.layer1(x)                                 # 32*14*14
        x = self.layer2(x)                                 # 64*7*7
        x = self.layer3(x)                                 # 128*7*7
        x = self.layer4(x)                                 # 256*7*7
        x = self.ca3(x) * x                                # 256*7*7
        x = self.sa3(x) * x                                # 256*7*7
        x = self.layer5(x)                                 # 256*3*3
        # print(x.shape)
        x = x.view(-1, 256 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        x = self.fc3(x)
        return x