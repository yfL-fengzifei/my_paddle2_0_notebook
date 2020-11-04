import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import DataLoader,Dataset
import numpy as np
from PIL import Image
import os

PATH=os.getcwd()
# path=os.path.join(PATH,'car_licence.zip')
# unzip_path=os.path.join(PATH,'car_licence')
# data_folders=os.listdir(unzip_path)
train_list_path=os.path.join(PATH,'train_data.list')
test_list_path=os.path.join(PATH,'test_data.list')
# print(train_list_path,test_list_path)

#解析数据
def parse_data(dataset_path):
    imgs = []
    labels = []
    with open(dataset_path, 'r',encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            img_path, label = line.split('\t')
            imgs.append(img_path)
            labels.append(int(label))

        assert len(imgs) == len(labels), "the number of imgs and labels not match"
    return imgs, labels

# 定义数据集
class CAR_LICENCE(paddle.io.Dataset):

    def __init__(self, data_path):
        self.imgs, self.labels = parse_data(data_path)

    def __getitem__(self, index):
        img = np.array(Image.open(self.imgs[index]).convert('L'), dtype='float32').reshape((1, 20, 20))
        label = np.array(self.labels[index])
        return img, label

    def __len__(self):
        return len(self.imgs)


# 实例化数据集
train_dataset = CAR_LICENCE(train_list_path)
test_dataset=CAR_LICENCE(test_list_path)
# print(train_dataset[0][0].shape,train_dataset[0][1].shape)

# len(train_dataset) #14506
# len(test_dataset) #1645

#实例化mini-batch
train_loader=paddle.io.DataLoader(train_dataset,batch_size=8,shuffle=True,drop_last=True)
test_loader=paddle.io.DataLoader(test_dataset,batch_size=8,shuffle=True,drop_last=True)

#构建网络
"""
version 1
"""
"""
# class MYLeNet(nn.Layer):
#     def __init__(self):
#         super(MYLeNet,self).__init__()

#         self.conv1=nn.Conv2D(1,28,5,1)
#         self.pool1=nn.MaxPool2D(2,1)
#         self.conv2=nn.Conv2D(28,32,3,1)
#         self.pool2=nn.MaxPool2D(2,1)
#         self.conv3=nn.Conv2D(32,32,3,1)
#         self.flatten=nn.Flatten()
#         self.linear=nn.Linear(32*10*10,65)
#         self.relu=nn.ReLU()

#         # self.softmax=nn.Softmax()

#     def forward(self,inputs):
#         x=self.conv1(inputs)
#         x=self.relu(x)
#         x=self.pool1(x)
#         x=self.relu(x)
#         x=self.conv2(x)
#         x=self.relu(x)
#         x=self.pool2(x)
#         x=self.conv3(x)
#         x=self.relu(x)
#         x=self.flatten(x)
#         x=self.linear(x)

#         return x
"""


"""
version 2
"""
class MYLeNet(nn.Layer):
    def __init__(self):
        super(MYLeNet, self).__init__()

        self.conv1 = nn.Conv2D(1, 28, 5, 1)
        self.pool1 = nn.MaxPool2D(2, 1)
        self.conv2 = nn.Conv2D(28, 32, 3, 1)
        self.pool2 = nn.MaxPool2D(2, 1)
        self.conv3 = nn.Conv2D(32, 32, 3, 1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(32 * 10 * 10, 65)
        # self.relu = nn.ReLU()

        # self.softmax=nn.Softmax()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(F.relu(x))
        x = self.conv2(x)
        x = self.pool2(F.relu(x))
        x = self.conv3(x)
        x = self.flatten(F.relu(x))
        x = self.linear(x)

        return x

#实例化网络
net=MYLeNet()
# paddle.summary(net,(1,1,20,20))

#训练
net.train()
epoch_num = 3

opt = paddle.optimizer.SGD(learning_rate=0.001, parameters=net.parameters())
loss_fn = nn.CrossEntropyLoss()  # 默认均值 原理是one-hot+logsoftmax
for epoch_id in range(epoch_num):
    for batch_id, data in enumerate(train_loader()):
        imgs, labels = data

        imgs = paddle.to_tensor(imgs)
        labels = paddle.to_tensor(labels.reshape((-1, 1)), dtype='int64')
        # print(imgs.shape,labels)

        preds = net(imgs)  # (N,num_class)
        # print(type(preds),preds.shape)

        loss = loss_fn(preds, labels)  # 得到的是一个batch的均值损失，因为默认是mean
        # loss=paddle.nn.CrossEntropyLoss(preds,labels)
        # print(type(loss),loss.shape,loss)

        acc = paddle.metric.accuracy(preds, labels)
        # print(type(acc),acc.shape,acc)

        if batch_id % 100 == 0:
            print('epoch:{},bacth:{},loss:{},acc:{}'.format(epoch_id, batch_id, loss.numpy(), acc.numpy()))

        loss.backward()
        opt.step()
        opt.clear_grad()

model_dict=net.state_dict()
opt_dict=opt.state_dict()
param_path=os.path.join(PATH,'net.pdparams')
opt_path=os.path.join(PATH,'net.pdopt')
paddle.save(model_dict,param_path)
paddle.save(opt_dict,opt_path)


