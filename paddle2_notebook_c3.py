"""
网络构建与训练
1.sequential组网 -简单顺序

2.subclass组网 -复杂网路：需要使用Layer子类定义的方式进行模型组网，即继承与nn.Layer

3.certain_backbone=paddle.vision.models.certain_backbone; #实例化网络

4.paddle.Model: 高级API，简化训练、评估、预测类代码开发，Net继承自paddle.nn.Layer,Model是指持有一个Net实例，同时指定损失函数，优化算法，评估指标的，可训练、评估、预测的实例，
model=paddle.Model(certain_net); #高级实例化 certain_net可以是certain_backbone或自构建net
model.prepare((paddle.optimizer=XXX,parameters=model.parameters()),paddle.nn.certain_loss,paddle.metric.certain_metric); #设置训练模型所需要的的optimizer,loss,metric
model.fit(certain_dataset,epochs,batchz_size,log_freq); #启动训练
model.evaluate(certain_dataset,log_freq,batch_size); #启动评估
model.summary((input)) #模型结构可视化
model.predict() #启动预测
"""

#GPU
# paddle.set_device('gpu')


#参数
"""
#model.parameters()
list
model.parameters()[0].numpy()/.name/.ndim/.dtype/.palce/.stop_gradient/.trainable

#model.named_parameters()
for name,param in model.named_parameters(): pass

#model.ceratin_layer
# model.certain_layer.parameters()
# model.certain_layer.weight/bias

#网络结构
model.summary()
paddle.summary()
"""

import paddle
import paddle.nn as nn

#模型构建
#1.sequnetial
minst=nn.Sequential(
    nn.Flatten(),
    nn.Linear(784,512),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(512,10),
)

#2.subclass
class MNIST(nn.Layer):
    def __init__(self):
        super(MNIST,self).__init__()
        self.flatten = paddle.nn.Flatten()
        self.linear_1 = paddle.nn.Linear(784, 512)
        self.linear_2 = paddle.nn.Linear(512, 10)
        self.relu = paddle.nn.ReLU()
        self.dropout = paddle.nn.Dropout(0.2)

    def forward(self,inputs):
        y = self.flatten(inputs)
        y = self.linear_1(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.linear_2(y)
        return y
model=MNIST()


#3.paddle.Model
train_dataset = paddle.vision.datasets.MNIST(mode='train')
test_dataset = paddle.vision.datasets.MNIST(mode='test')
lenet = paddle.vision.models.LeNet()
model2=paddle.Model(lenet)


#模型训练
#基础-伪代码
"""
datasets=
net=
dataloader=paddle.io.DataLoader(datasets,...)
epoch_num
opt=paddle.optimizer.certain_opt
for epoch in range(epoch_num):
    for id,data in enumerate(dataloader()):
        x,y=data
        pred=net(x)
        loss=paddle.nn.functional.certain_loss(pred,y,...)
        acc=paddle.metric.certain_metric(pred,y,...)
        avg_acc=paddle.mean(acc)
        loss.backward()
        opt.step()
        opt.clear_grad()
"""

#高级-伪代码
"""
datasets=
net=
model=paddle.Model(net)
model.prepare(
              (paddle.opt.certain_opt,parameters=model.parameters()),
              paddle.nn.certain_loss(),
              paddle.metric.certain_metric())
model.fit(datasets,batch_size,...) #还可以传入dataloader
model.evaluate(datasets,batch_size,...)
model.predict()
"""

#paddle.summary()
"""
两种方法
model.summary(XXX) 对应于上述高级API
paddle.summary(net,XXX) 对应于基础API
"""
class LeNet(nn.Layer):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2D(
                1, 6, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2D(2, 2),
            nn.Conv2D(
                6, 16, 5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2D(2, 2))

        if num_classes > 0:
            self.fc = nn.Sequential(
                nn.Linear(400, 120),
                nn.Linear(120, 84),
                nn.Linear(
                    84, 10))

    def forward(self, inputs):
        x = self.features(inputs)

        if self.num_classes > 0:
            x = paddle.flatten(x, 1)
            x = self.fc(x)
        return x

lenet = LeNet()

params_info = paddle.summary(lenet, (1, 1, 28, 28))
print(params_info)

nn.MaxPool2D(2,)

