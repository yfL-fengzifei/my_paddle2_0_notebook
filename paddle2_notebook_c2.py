"""
数据集
"""

"""
加载数据集
datatsets=XX #paddle.vision.datasets
dataloader=paddle.io.DataLoader(datasets) #paddle.io.dataloader

自定义数据集仅dataset,简单定义后for data,label in train_dataset: pass

数据预处理
paddle.vision.transformas
transforms.compose #paddle.vision.datasets.certain_datasets(...,transform=transforms)
"""

import paddle
from paddle.io import Dataset
import paddle.nn as nn
import numpy as np

#查看数据集
print(paddle.vision.datasets.__all__)

# class MyDataset(Dataset):
#     #第一步：继承paddle.io.Dataset
#     def __init__(self,model='train'):
#         #第二步：实现构造函数，定义数据读取方式，划分训练和测试数据集
#         super(MyDataset, self).__init__()
#
#         if model=='train':
#             self.data=[
#                 ['traindata1', 'label1'],
#                 ['traindata2', 'label2'],
#                 ['traindata3', 'label3'],
#                 ['traindata4', 'label4'],
#             ]
#         else:
#             self.data=[
#                 ['testdata1', 'label1'],
#                 ['testdata2', 'label2'],
#                 ['testdata3', 'label3'],
#                 ['testdata4', 'label4'],
#             ]
#
#     def __getitem__(self,index):
#         #第三步：实现_getitem_方法，定义指定index时如何获取数据，并返回单条数据（训练数据，对应数据标签）
#         data=self.data[index][0]
#         label=self.data[index][1]
#
#         return data,label
#
#     def __len__(self):
#         #第四步：实现__len__方法，返回数据集总数目
#         return len(self.data)
#
# #测试
# train_dataset=MyDataset(model='train')
# val_dataset=MyDataset(model='test')
# print('train dataset')
# for data,label in train_dataset:
#     print(data,label)
#
# print('val dataset')
# for data,label in train_dataset:
#     print(data,label)

