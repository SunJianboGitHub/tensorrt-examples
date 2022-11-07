# 如何导出简单准确的ONNX模型
    - 主要是去掉gather、shape类的节点
    - 有时候，不去修改这些节点好像也没什么问题，但是需求复杂后依旧会出现各类问题，因此我们倾向于尽可能去除这些节点

# 常用的去除节点、简化模型的方法有
    - 对于任何用到shape和size返回值的参数时，避免直接使用tesnsor.size的返回值，而是加上int转换为Python类型，
      比如，将tensor.view(tensor.size(0), -1)这类操作修改为tensor.view(int(tensor.size(0)), -1)
    - 对于nn.Upsample或nn.functional.interpolate函数,使用scale_factor指定倍率，而不是使用size参数指定
    - 对于reshape、view操作时，-1的指定请放在batch维度。其它维度可以计算出来，batch维度进制指定为大于-1的明确数字
    - torch.onnx.export指定dynamic_axes参数，并且只指定batch维度，不指定其它维度。我们只需要动态batch，相对动态宽高有其它方案。
    - 尽量不导出多个维度(5个以上的维度)













































