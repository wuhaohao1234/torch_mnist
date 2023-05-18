from model.model import LeNet, AlexNet, AlexNet_CBAM, New_AlexNet_CBAM
import torch
from torchvision.datasets import mnist
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.nn import CrossEntropyLoss
import json

def test(test_dataloader, model, criterion):                # 测试每个epoch
    print("testing:")
    model.eval()                                            # 测试时候需要 model.eval()
    total_correct = 0
    total_num = 0
    total_loss = 0.0
    iters = 0
    with torch.no_grad():                                   # 测试时候不记录梯度,节省算力，加快速度
        for data in tqdm(test_dataloader):                  # 下面就跟训练是一样了
            imgs, labels = data
            bs = imgs.shape[0]

            outputs = model(imgs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)   
            total_correct += (predicted == labels).sum().item()
            total_num += bs
            total_loss += loss.item()
            iters += 1
        
        test_acc = total_correct/total_num
        test_loss = total_loss/iters

        return test_acc, test_loss

if __name__ == '__main__':
    batch_size = 128  # 一个一个测

    train_dataset = mnist.MNIST(root='./DataSet', train=True, transform=ToTensor(),download=True)
    test_dataset = mnist.MNIST(root='./DataSet', train=False, transform=ToTensor(),download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,drop_last=False)
    cross_error = CrossEntropyLoss()


    model_name = 'alexnet'

    model = AlexNet()
    # if model_name == 'lenet':
    #     model = LeNet()
    # elif model_name == 'alexnet':
    #     model = AlexNet()
    # elif model_name == 'alexnet_cbam':
    #     model = AlexNet_CBAM()
    # elif model_name == 'new_alexnet_cbam':
    #     model = New_AlexNet_CBAM()


    model.load_state_dict(torch.load(f'model/{model_name}.pth'))

    alexnet_train_acc, train_loss = test(train_loader, model, cross_error)
    alexnet_train_loss, test_loss = test(test_loader, model, cross_error)       # 测试
    # ---------------------------------------------------------------------------------------------
    model_name = 'alexnet_cbam'

    model = AlexNet_CBAM()

    model.load_state_dict(torch.load(f'model/{model_name}.pth'))

    alexnet_cbam_train_acc, train_loss = test(train_loader, model, cross_error)
    alexnet_cbam_train_loss, test_loss = test(test_loader, model, cross_error)       # 测试

    # ---------------------------------------------------------------------------------------------
    model_name = 'alexnet_cbam'

    model = AlexNet_CBAM()

    model.load_state_dict(torch.load(f'model/{model_name}.pth'))

    new_alexnet_cbam_train_acc, train_loss = test(train_loader, model, cross_error)
    new_alexnet_cbam_train_loss, test_loss = test(test_loader, model, cross_error)       # 测试

    train_json = {
        "alexnet_train_acc" : alexnet_train_acc * 100,
        "alexnet_train_loss" : alexnet_train_loss * 100,
        "alexnet_cbam_train_acc" : alexnet_cbam_train_acc * 100,
        "alexnet_cbam_train_loss" : alexnet_cbam_train_loss * 100,
        "new_alexnet_cbam_train_acc" : new_alexnet_cbam_train_acc * 100,
        "new_alexnet_cbam_train_loss" : new_alexnet_cbam_train_loss * 100
    }
    with open("train.json", "w") as json_file:
        json.dump(train_json, json_file)

    print(f'【{model_name}】：',end='\n')
    # print(f'train acc: {train_acc}, train loss: {train_loss}')
    # print(f'test acc: {test_acc}, test loss: {test_loss}')