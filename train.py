from model.model import LeNet, AlexNet, AlexNet_CBAM, New_AlexNet_CBAM
import torch
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, RandomHorizontalFlip, RandomRotation, Compose
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random

#当使用gpu训练模型时，可能引入额外的随机源，使得结果不能准确再现（gpu提供了多核并行计算的基础结构），保证每次运行网络的时候相同输入的输出是固定
# torch.manual_seed(1)
torch.cuda.manual_seed(1)                  # 为GPU设置种子，生成随机数
np.random.seed(1)
random.seed(1)
torch.backends.cudnn.deterministic = True  # 为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。（输入形状和网络结构固定）



def train_epoch(trian_dataloader, model, criterion, i, epochs):     #训练每个epoch
    model.train()
    print("Epoch %d/%d:" % (i + 1, epochs))
    total_loss = 0.0
    total_correct = 0.0
    total_num = 0
    iters = 0
    for imgs, labels in tqdm(trian_dataloader):                    #读取图像和标签进行训练
        imgs = imgs.cuda()                                         #.cuda()是将数据放到gpu上,以提高运算效率
        labels = labels.cuda()
        bs = labels.shape[0]                                       #128  bs是batch_size大小，就是每次处理的图片数目
        outputs = model(imgs)                                      #得到输出 onehot编码 [1*10]
        loss = criterion(outputs, labels)                          #计算交叉熵loss
        _, predicted = torch.max(outputs.data, 1)                  #获得最大的下标就是预测的标签，下标介于 0~9
        optimizer.zero_grad()
        loss.backward()                                            #反向传播，求取梯度
        optimizer.step()                                           #根据梯度更新网络参数
        correct_num = (predicted == labels).sum().item()           #计算准确的个数
        total_correct += correct_num
        total_loss += loss.item()
        total_num += bs
        iters += 1

    train_loss = total_loss / iters                               #总loss
    train_acc = total_correct / total_num                         #准确率
    print('\ntrain loss', train_loss, end="")
    print('    train acc:', train_acc)
    return train_acc, train_loss


def test(test_dataloader, model, criterion, best_acc, model_name):   #测试每个epoch
    print("testing:")
    model.eval()                                                     #测试时候需要 model.eval()
    total_correct = 0.0
    total_num = 0
    total_loss = 0.0
    iters = 0
    with torch.no_grad():                       #叫停autograd模块，节省GPU算力和显存，测试时候不记录梯度
        for data in tqdm(test_dataloader):      #下面就跟训练是一样了
            imgs, labels = data
            bs = imgs.shape[0]
            imgs = imgs.cuda()
            labels = labels.cuda()

            outputs = model(imgs)
            loss = criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)   
            total_correct += (predicted == labels).sum().item()
            total_loss += loss
            total_num += bs
            iters += 1
        
        test_acc = total_correct/total_num
        test_loss = total_loss/iters

        if test_acc > best_acc:                #保存准确率最高的模型
            best_acc = test_acc
            torch.save(model.state_dict(), f'model/{model_name}.pth')

        print("test loss: ", test_loss, end="")
        print(" test acc: ",test_acc)
        print("best acc now: ", best_acc)
        return test_acc, test_loss, best_acc


if __name__ == '__main__':
    batch_size = 128
    train_dataset = mnist.MNIST(root='./DataSet', train=True, transform=Compose([
                                                                                # RandomHorizontalFlip(0.5),
                                                                                 RandomRotation(30),
                                                                                 ToTensor()]), download=True)      #训练集
    test_dataset = mnist.MNIST(root='./DataSet', train=False, transform=ToTensor(),download=True)      #测试集
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,drop_last=False)

    model_name = 'new_alexnet_cbam'

    if model_name == 'lenet':
        model = LeNet()
    elif model_name == 'alexnet':
        model = AlexNet()
    elif model_name == 'alexnet_cbam':
        model = AlexNet_CBAM()
    elif model_name == 'new_alexnet_cbam':
        model = New_AlexNet_CBAM()

    model = model.cuda()

    optimizer = Adam(model.parameters(), lr=3e-4)
    cross_error = CrossEntropyLoss()
    epochs = 100

    best_acc = 0.0
    train_acc_record = []
    train_loss_record = []
    test_acc_record = []
    test_loss_record = []


    for i in range(0,epochs):
        train_acc, train_loss = train_epoch(train_loader, model, cross_error, i, epochs)               #训练集
        test_acc, test_loss, best_acc = test(test_loader, model, cross_error, best_acc, model_name)    #测试集
        train_acc_record.append(train_acc)
        train_loss_record.append(train_loss)
        test_acc_record.append(test_acc)
        test_loss_record.append(test_loss)
        if i == 60:
            optimizer.param_groups[0]['lr'] *= 0.5
        if i == 80:
            optimizer.param_groups[0]['lr'] *= 0.5


    # 画acc曲线
    plt.figure(1)
    plt.plot(range(epochs), train_acc_record, label='train_acc')
    plt.plot(range(epochs), test_acc_record, label='test_acc')
    plt.title(f"{model_name}'s Accuracy curve")
    plt.legend()
    plt.savefig(f'./static/acc_{model_name}.png',bbox_inches='tight')


    # 画loss曲线
    plt.figure(2)
    plt.plot(range(epochs), train_loss_record, label='train_loss')
    plt.plot(range(epochs), test_loss_record, label='test_loss')
    plt.title(f"{model_name}'s Loss curve")
    plt.legend()
    plt.savefig(f'./static/loss_{model_name}.png',bbox_inches='tight')

    print('Done!')