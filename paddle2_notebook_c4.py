""""
模型存储载入-高级API

Paddle保存的模型有两种格式，
一种是训练格式，保存模型参数和优化器相关的状态，可用于恢复训练；
一种是预测格式，保存预测的静态图网络结构以及参数，用于预测部署。

#高级API
paddle.Model.fit(训练接口，同时带有参数存储得到功能)，model.fit也可以保存模型，只保存模型参数不保存优化器参数，每个epoch后只生成一个.pdparams文件，可以边训练边保存
paddle.Model.save ,可以保存模型结构、网络参数、优化器参数
paddle.Model.load

#开最终要得到什么，选择哪种保存方式，不同的保存方式对应于不同的网络构建和训练
paddle.Model.save(trianing=True) 用于训练调优，在训练过程中使用，保存网络参数和优化器参数，生成pdparams和paopt文件，但是只会在整个模型训练完成后生成所有epoch参数文件，
paddle.Model.save(training=False) 用于训练部署，表明训练已经结束，存储的是预测模型结构和网络模型参数
"""

"""================================================================="""
"""
模型封装 https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-rc/tutorial/quick_start/high_level_api/high_level_api.html

## 场景1：动态图模式
## 1.1 为模型预测部署场景进行模型训练
## 需要添加input和label数据描述，否则会导致使用model.save(training=False)保存的预测模型在使用时出错
input = paddle.static.InputSpec([None, 1, 28, 28], dtype='float32')
label = paddle.static.InputSpec([None, 1], dtype='int8')
model = paddle.Model(mnist, input, label)

## 1.2 面向实验而进行的模型训练
## 可以不传递input和label信息
# model = paddle.Model(mnist)
"""

"""
高层API
model = paddle.Model(Mnist())
# 预测格式，保存的模型可用于预测部署
model.save('mnist', training=False)
# 保存后可以得到预测部署所需要的模型

model.save() 用于后续模型的Fine-tuning（接口参数training=True）或推理部署（接口参数training=False）在动态图模式训练时保存推理模型的参数文件和模型文件，需要在forward成员函数上添加@paddle.jit.to_static装饰器 #启动推理
"""


"""================================================================="""
#面向实验
"""
## 1.2 面向实验而进行的模型训练
## 可以不传递input和label信息
# model = paddle.Model(mnist)

#train pass
model.prepare()
model.fit()
model.evalue()

model.save() 用于后续模型的Fine-tuning（接口参数training=True）
"""

#面向预测部署
"""
## 1.1 为模型预测部署场景进行模型训练  参见'paddle2_notebook_c4.py'
## 需要添加input和label数据描述，否则会导致使用model.save(training=False)保存的预测模型在使用时出错
input = paddle.static.InputSpec([None, 1, 28, 28], dtype='float32')
label = paddle.static.InputSpec([None, 1], dtype='int8')
model = paddle.Model(mnist, input, label)

# model = paddle.Model(Mnist())
# # 预测格式，保存的模型可用于预测部署 ???所以这里有点问题 

model.save('mnist', training=False)
# 保存后可以得到预测部署所需要的模型

model.save()推理部署（接口参数training=False）在动态图模式训练时保存推理模型的参数文件和模型文件，需要在forward成员函数上添加@paddle.jit.to_static装饰器 #启动推理 (这对应于用自定义网络构建中)
"""

