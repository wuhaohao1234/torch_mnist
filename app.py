import re
import torch
import base64            #使用base64编解码
from flask import Flask, render_template, request, jsonify,session, redirect, url_for, flash
import json
import numpy as np
from PIL import Image                           #图像处理库
import torchvision.transforms as transforms     #图像预处理，定义了一系列数据转换形式，还能对数据进行处理
from model.model import LeNet, AlexNet, AlexNet_CBAM, New_AlexNet_CBAM



app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    global model
    # 定义选项列表
    options1 = ['AlexNet', 'AlexNet_CBAM', 'New_AlexNet+CBMA']
    options2 = ['acc', 'loss']
    filename = 'acc_lenet.png'
    radio = 'AlexNet'
    selected1 = options1[0]
    selected2 = options2[0]

    # 获取用户选择的值
    if request.method == 'POST':
        selected1 = request.form.get('dropdown1')
        selected2 = request.form.get('dropdown2')
        radio = request.form.get('radio')

    # 根据用户选择的值决定要展示的图像
    if selected1 == 'AlexNet' and selected2 == 'acc':
        filename = 'acc_alexnet.png'
    elif selected1 == 'AlexNet_CBAM' and selected2 == 'acc':
        filename = 'acc_alexnet_cbam.png'
    elif selected1 == 'New_AlexNet+CBMA' and selected2 == 'acc':
        filename = 'acc_new_alexnet_cbam.png'
    elif selected1 == 'AlexNet' and selected2 == 'loss':
        filename = 'loss_alexnet.png'
    elif selected1 == 'AlexNet_CBAM' and selected2 == 'loss':
        filename = 'loss_alexnet_cbam.png'
    elif selected1 == 'New_AlexNet+CBMA' and selected2 == 'loss':
        filename = 'loss_new_alexnet_cbam.png'
    # 根据用户选择的值决定要使用的模型
    if radio == 'AlexNet':
        model = AlexNet()
        model.load_state_dict(torch.load(f'model/alexnet.pth'))
    elif radio == 'AlexNet+CBAM':
        model = AlexNet_CBAM()
        model.load_state_dict(torch.load(f'model/alexnet_cbam.pth'))
    elif radio == 'new_AlexNet+CBMA':
        model = New_AlexNet_CBAM()
        model.load_state_dict(torch.load(f'model/new_alexnet_cbam.pth'))

    print(' ------------------ ')
    print("【selected1】: {}".format(selected1))
    print("【selected2】: {}".format(selected2))
    print("【filename】: {}".format(filename))
    print("【select model】: {}".format(radio))
    print(' ------------------ ')

    # 如果有选择，则呈现图像
    if filename:
        # image_url = f"/static/{filename}"
        image_url = filename
        return render_template('index.html', show_results=True, static_folder='static', image_url=image_url, options1=options1, options2=options2, selected1=selected1, selected2=selected2, radio=radio)
    else:
        print("no filename!!!")
        return render_template('index.html',  show_results=False, static_folder='static',  options1=options1, options2=options2, radio=radio)




@app.route('/predict/', methods=['Get', 'POST'])
def preditc():
    global model                       #引用全局变量model
    parseImage(request.get_data())     #对获取的图片进行解析

    # print(request.get_data())

    '''预测'''
    data_transform = transforms.Compose([transforms.ToTensor(), ])
    root = 'static/output.png'
    img = Image.open(root)
    img = img.resize((28,28))
    # print(len(img.split()))

    img = img.convert('L')               #转为灰度图像，降低通道数到1
    # print(len(img.split()))

    # print(type(img))                   #‘PIL.Image.Image’
    img = data_transform(img)
    print(img.shape)
    img = torch.unsqueeze(img, dim=0)   # 输入要与model对应

    print(img.shape)
    predict_y = model(img.float()).detach()


    predict_ys = np.argmax(predict_y, axis=-1)
    ans = predict_ys.item()
    # print(predict_y)
    # print(predict_y.numpy().squeeze()[ans])
    digi0 = predict_y.numpy().astype(float).squeeze()[0]
    digi1 = predict_y.numpy().astype(float).squeeze()[1]
    digi2 = predict_y.numpy().astype(float).squeeze()[2]
    digi3 = predict_y.numpy().astype(float).squeeze()[3]
    digi4 = predict_y.numpy().astype(float).squeeze()[4]
    digi5 = predict_y.numpy().astype(float).squeeze()[5]
    digi6 = predict_y.numpy().astype(float).squeeze()[6]
    digi7 = predict_y.numpy().astype(float).squeeze()[7]
    digi8 = predict_y.numpy().astype(float).squeeze()[8]
    digi9 = predict_y.numpy().astype(float).squeeze()[9]
    return jsonify(ans, digi0, digi1, digi2, digi3, digi4, digi5, digi6, digi7, digi8, digi9)

@app.route('/train/', methods=['Get', 'POST'])
def train():
    # alexnet_train_acc = 99.95
    # alexnet_train_loss = 99.33
    # alexnet_cbam_train_acc = 99.46
    # alexnet_cbam_train_loss = 99.48
    # new_alexnet_cbam_train_acc = 99.99
    # new_alexnet_cbam_train_loss = 99.61
    with open('train.json', 'r') as file:
        # 读取JSON数据
        data = json.load(file)
    return jsonify(data)


def parseImage(imgData):
    imgStr = re.search(b'base64,(.*)', imgData).group(1)
    with open('./static/output.png', 'wb') as output:            #使用with进行文件上下文管理
        output.write(base64.decodebytes(imgStr))

if __name__ == '__main__':
    app.run(debug=True)