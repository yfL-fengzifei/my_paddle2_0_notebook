"""
模型存储载入-基础API
"""
"""
#参数存储(训练调优)
paddle.save
paddle.load
--> state_dict 
path.pdparams
path.pdopt
<--
paddle.load(有模型代码，仅载入参数)

#模型参数存储(训练部署)
paddle.jit.save
padle.jit.load
--> inference_model
path.model
path.pdiparams
path.pdiparams.info
<--
paddle.load 或
paddle.jit.load(无模型代码，载入模型和参数)

#高级API
paddle.Model.fit(训练接口，同时带有参数存储得到功能)
paddle.Model.save
paddle.Model.load
"""

#伪代码-paddle.save/load (训练调优)
"""仅存储参数"""
"""
layer_dict=layer.state_dict()
opt_dict=opt.state_dict()
paddle.save(layer_dict,'net.pdparams')
paddle.save(opt_dict,'net.pdopt')

layer_state_dict=paddle.load('net.pdparams')
opt_state_dict=paddle.load('opt.pdparams')
layer=certain_net()
layer.set_state_dict(layer_state_dict)
opt.set_state_dict(opt_state_dict)
"""

"""====================================================================================================="""

#伪代码-paddle.jit.save/load (训练部署)
"""同时存储网络结构和参数"""

#动转静训练+模型参数存储 见'paddle2_notebook_c5.py'
"""
动转静训练相比直接使用动态图训练具有更好的执行性能，训练完成后，直接将目标Layer传入 paddle.jit.save 存储即可

import paddle
from paddle.jit import to_static
from paddle.static import InputSpec

改动一：forward方法需要由paddle.jit.to_static装饰，经过装饰后，相应layer在执行时，会先生成描述模型的program,然后通过执行program获取计算结果(或@paddle.jit.to_static(input_spec=[InputSpec(shape=[None, 784], dtype='float32')]))
class Net(nn.Layer):
    def __int__():
        pass
        
    @paddle.jit.to_static
    def forward(self,x):
        pass
train()
    pass

#save
path='certain_path'
paddle.jit.save(layer,path)  

"""

#动态图训练+模型参数存储
"""
动态图模式相比动转静模式更加便于调试，如果您仍需要使用动态图直接训练，也可以在动态图训练完成后调用 paddle.jit.save 直接存储模型和参数
与动转静训练相比
forward方法不需要额外装饰
paddle.jit.save 需要指定layer的InputSpec


path='certain_path'
paddle.jit.save(layer=layer,path=path,input_spec=[InputSpec(shape=[None, 784], dtype='float32')])
"""

#paddle.jit.load
"""
#同时加载模型和参数
path='certain_path'
loaded_layer=paddle.jit.load(path)
#预测推理
loaded_layer.eval()
pred=loaded_layer(x)

#fine_tune见文档

#仅加载模型(还是用paddle.jit.save保存的)
layer=certain_layer()
path='certain_path'
state_dict=paddle.load(path)
#预测推理
layer.set_state_dict(state_dict,xxx)
layer.eval()
pre=layer(x)

"""



