import json
import torch
import torch.nn as nn


'''
    1. 实现TensorRT算子的第一步, 导出合适的ONNX模型
    2. 必须实现一些特殊的方法,才可以导出正确的ONNX模型,并用于TensorRT
    3. 下面展示了一个简单的插件的导出方式

'''


class SwishAndBiasImpl(torch.autograd.Function):
    
    @staticmethod
    def symbolic(g, input, bias):
        # name_s，在g.op这个场景下，有两个作用
        # 1. 告诉g.op，这个是属性
        # 2. 告诉g.op，这个属性的名称是name，类型是s，也就是string
        #
        # 老师提供的tensorRT框架插件做了定义
        # 1. 名称必须是Plugin
        # 2. name_s指定插件的子名称
        # 3. info_s指定需要带进去的属性信息
        #
        # 如果按照官方的写法，那么插件名称可以任意，属性可以任意
        # 即不需要一定写name_s，可以任意写
    
        return g.op("Plugin", input, bias, name_s="Swish", info_s=json.dumps(
            {
                "size": 224*224,
                "shape": [1, 3, 224, 224],
                "module": "abcdef"
            }
        ))
    
    
    @staticmethod
    def forward(context, input, bias):
        # 告诉pytorch，这个input需要保留到反向传播时使用
        # 如果不告诉，则input会在节点forward后内存被回收利用
        # 这是因为内存高效复用的一个原则
        context.save_for_backward(input)
        return input * torch.sigmoid(input) + bias
    
    @staticmethod
    def backward(context, grad_output):
        input = context.saved_tensors[0]
        sigmoid_input = torch.sigmoid(input)
        # 返回的第一个tensor，是input的导数
        # 返回的第二个tensor，是bias的导数
        return grad_output * (sigmoid_input * (1 + input * (1 - sigmoid_input))), grad_output
    
    

# 实际的swish算子的实现类
# 这里其实是随便模拟一个网络层,偏置参数是想设置一个带参数的层,类似于卷积层
class EfficientSwish(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.full((1,), 3.15))
        
    def forward(self, x):
        return SwishAndBiasImpl.apply(x, self.bias)



# 基于自己实现的swish算子，构建网络模型
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv = nn.Conv2d(1, 1, 3, stride=1, padding=0, bias=False)
        self.conv.weight.data = torch.FloatTensor([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]).view(1, 1, 3, 3)
        
        self.swish = EfficientSwish()

    def forward(self, x):
        return self.swish(self.conv(x))



input = torch.arange(9).view(1, 1, 3, 3).float()
print(input)


# 搭建模型
model = Model()


# model.train()
# y = model(input)
# loss = y.mean()
# loss.backward()

# print(loss)
# print(model.swish.bias.grad)



# 开始准备导出ONNX模型
model.eval()

y = model(input)
print(y)




torch.onnx.export(
    model,
    (input, ),
    "./plugin_SwishAndBias.onnx",
    opset_version=11,                                   # 导出的版本
    verbose=True,                                       # 是否打印导出过程中的详细信息
    enable_onnx_checker=False                           # 导出过程中是否检查算子是否支持, 如果有自定义算子, 应设置为FALSE
)







