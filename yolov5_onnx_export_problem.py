#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   export_yolov5_onnx.py
@Time    :   2022/11/07 10:41:42
@Author  :   JianBo Sun 
@Version :   1.0
@License :   (C)Copyright 2021-2022, ai-uav-cyy-cmii
@Desc    :   yolov5导出onnx,然后转换为TensorRT,会遇到的问题
'''

# yolov5转TensorRT模型,可能会遇到的问题
# yolov5 -> onnx -> tensorRT

'''
    1. 问题1: Slice节点切片问题, 提示slice is out of range
        - 原因：
            - slice节点,是由YOLO中的Focus层导出所致,生成onnx时,其ends值【通常是-1】给定为极大整数值,导致两者不兼容
        - 解决方案:
            - 修改pytorch的导出代码,让这个ends值是tensorRT合理的.
            - 修改方式是,修改opset_version中找到对应的节点,也就是/site-packages/torch/onnx/symbolic_opset11.py文件中slice函数
            - 干掉Focus层,使用cuda去实现.把Focus认为是预处理层,与BGR->RGB转换、Normalize进行合并为一个操作
        
    2. 问题2: model.model[-1].export=True,指定为导出模式
        - model.model[-1]是所有layer的Sequential结构,model.model[-1]是指最后一层,即Detect层
        - model.model[-1].export谁知为True,会使得Detect在forward时,返回原始数据,而不进行sigmoid等复原操作.
        - 因为复原操作需要我们自己定义在cuda核中,作为后处理实现
        - 后处理干的事情就是把网络输出结果恢复成图像大小的框
        - 整个推理过程就是, 预处理(BRG2RGB/Focus/Noramlize/center_align) -> CNN推理(TensorRT) -> 后处理(Decode成边界框)

    3. Gather的错误,while parsing node number 97 [Gather]:
        - 原因:
            - pytorch/onnx/tensorRT之间没有统一,尽量去除所有的Gather节点
        - 解决方案:
            - 分析原因,干掉Gather
            
                - 出现Gather的第一个场景: Resize节点
                    - 修改 def symbolic_fn(g, input, output_size, *args):函数的返回值为: return g.op("Upsample", input, scales_f=scales)
                    - 其中的scales = [1, 2, 2], 分别表示 channel, height, width, 的参数
                    - 这里的scales是指Upsample的缩放倍数
                    - Gather是由symbolic_fn内的调用造成的,经过这个操作后,Resize以及各种Gather操作合并为一个Upsample,去掉Gather
                    
                - 出现Gather的第二个场景: Reshape和Transpose节点
                    - 出现在网络的输出节点上,Detect模块上
                    - 由于Detect的forward中,使用view函数,输入的参数是来自x[i].shape。shape会在导出onnx时进行跟踪并生成节点
                      因此产生了Gather、shape、concat、constant等一系列多余节点,
                      将x[i].shape返回值,强制转换为int(python类型)时,可以避免节点跟踪和生成多余节点
    
    4. 维度问题,onnx和pytorch导出可以是5个维度,但是TensorRT显示的是4个维度
    
        - 原因：
            - 目前框架内使用的是TensorRT的3个维度的版本(CHW),N是由用户推理指定(MaxBatchSize,也是enqueue对应的Batch参数) 
            - 目前没有考虑5个维度的情况,因此需要去掉5个维度的问题,通常这都不是问题
            - 如果使用多维度(5个维度),灵活度提高了,但是复杂度会异常高
            
        - 解决方案:
            - 去掉5个维度的影响,这通常都是可以去掉的,在yolov5中,去掉reshape和transpose(也就是view和permute)
            - 输出维度为 (nc + 5) * 3, height, width     [C, H, W]   
'''

















































