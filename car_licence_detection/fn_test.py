import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import DataLoader,Dataset
import numpy as np
from PIL import Image
import os

# #验证corss_entropy and acc
# input_data=np.random.random((5,5)).astype('float64')
# # input_data.shape,input_data.dtype
# label_data=np.random.randint(0,5,size=(5)).astype('int64')
# # label_data.shape,label_data.dtype
# input_data=paddle.to_tensor(input_data)
# # label_data=paddle.to_tensor(label_data)
# label_data=paddle.to_tensor(label_data.reshape((5,1)))
# # input_data.shape,label_data.shape

# # # weight_data=np.random.random([100]).astype('int64')
# # # weight_data.shape

# # # loss=paddle.nn.CrossEntropyLoss()
# # loss=paddle.nn.CrossEntropyLoss(reduction='none')
# # logsoftmax=nn.LogSoftmax()
# # # nllloss=nn.NLLLoss()
# # nllloss=nn.NLLLoss(reduction='none')

# # pred_data=loss(input_data,label_data)
# # pred_data2_logsoftmax=-logsoftmax(input_data)
# # # print('2:',pred_data2_logsoftmax.shape)
# # pred_data2=nllloss(pred_data2_logsoftmax,label_data)

# # print('pred:')
# # print(pred_data2_logsoftmax)
# # print('label:')
# # print(label_data)
# # print('loss:')
# # print(pred_data)
# # print('loss vil:')
# # print(pred_data2)
# # print('mean loss:')
# # print(paddle.mean(pred_data2))

# # loss_mean=nn.CrossEntropyLoss(reduction='mean')
# # pred_data_mean=loss_mean(input_data,label_data)
# # nllloss_mean=nn.NLLLoss(reduction='mean')
# # pred_data2_mean=nllloss_mean(pred_data2_logsoftmax,label_data)
# # print('loss mean:')
# # print(pred_data_mean)
# # print('loss vil:')
# # print(pred_data2_mean)


# # input_data_exp=paddle.exp(input_data)
# # input_data_exp_sum=paddle.sum(input_data_exp,axis=-1,keepdim=True)
# # input_data_exp_softmax=input_data_exp/input_data_exp_sum
# # input_data_exp_logsoftmax=paddle.log(input_data_exp_softmax)
# # # input_data_exp_sum.shape,input_data_exp.shape,input_data_exp_logsoftmax.shape
# # # print(pred_data2_logsoftmax==input_data_exp_logsoftmax)

# # top_=paddle.topk(input_data,2)
# # print(input_data)
# # print(type(top_),top_)
# # print(label_data)


# # acc=paddle.metric.accuracy(input_data,label_data)
# # print(acc)