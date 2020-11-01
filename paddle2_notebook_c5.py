"""
动静态图
动态图训练的模型，可以通过动静转换功能，转换为可部署的静态图模型
"""
import paddle
from paddle.jit import to_static
from paddle.static import InputSpec

class SimpleNet(paddle.nn.Layer):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.linear = paddle.nn.Linear(10, 3)

    #第一处改动
    #通过InputSepc指定输入数据的形状，None表示可变长
    #通过to_static装饰器将动态图转换为静态图program
    @to_static(input_spec=
               [InputSpec(shape=[None,10],name='x'),
                InputSpec(shape=[3],name='y')])
    def forward(self,x,y):
        out=self.linear(x)
        out=out+y
        return out

net=SimpleNet()

#第二处改动，保存静态图模型，可用于预测部署
paddle.jit.save(net,'./simple_net')
