#include <stdio.h>
#include <iostream>
#include <NvInfer.h>
#include <cuda_runtime.h>
# include <opencv2/core.hpp>
# include <opencv2/opencv.hpp>

#include "filetools.hpp"


/*
 * 用TensorRT实现一个CNN推理，模型是手动构建的方式
 * 预期网络具有4个节点
 * input节点，为输入
 * conv1节点，为卷积，其输入是input
 * relu1节点，为激活，其输入是conv1
 * output节点，为输出，其输入是relu1
 * 
*/


// 1.定义Logger类
//     TensorRT内部出现的任何消息，都会通过Logger类打印出来
//     根据消息等级区分问题，有助于排查bug
// 纯虚函数
//     a.具有纯虚函数的类，不能够被实例化
//     b.纯虚函数是只有声明没有实现的虚函数，也就是尾巴后面给 =0
//     c.作用是，声明函数，实现交给子类，也就是常说的接口类
//     d.如果子类继承的基类中存在纯虚函数，如果需要实例化子类，则必须实现纯虚函数
class JLogger: public nvinfer1::ILogger
{
    public:
        virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override
        {
            printf("LOG[%d]: %s\n", severity, msg);
        }
};




void model_build()
{
    // 2.实例化Logger，作为全局日志打印的东西
    JLogger logger;

    // 3.构建模型编译器
    //  - 创建网络
    //  - 创建编译配置
    auto builder = nvinfer1::createInferBuilder(logger);
    uint32_t flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = builder->createNetworkV2(flag);                      // 设置显示的batch
    auto config = builder->createBuilderConfig();
    
    // 配置网络参数
    // 配置最大的batchsize，意味着推理所指定的batch参数不能超过这个
    builder->setMaxBatchSize(1);

    // 配置工作空间的大小
    // 每个节点不用自己管理内存空间，不用自己去cudaMolloc显存
    // 使得所有节点均把workspace当做显存池，重复使用，使得内存更加紧凑、高效
    // 1 << 30 大概是1G
    config->setMaxWorkspaceSize(1 << 30);

    // 默认情况下，使用的是FP32推理，如果希望使用FP16，可以设置这个flags
    // builder->platformHasFastFp16(); 这个函数告诉你，当前显卡是否具有fp16加速的能力
    // builder->platformHasFastInt8();  这个函数告诉你，当前显卡是否具有int8的加速能力
    // builder->setFp16Mode(true);  以前的写法
    // 如果要使用int8，则需要做精度标定
    // 模型量化内容，把你的权重变为int8格式，计算乘法。减少浮点数乘法操作，用整数(int8)来替代
    config->setFlag(nvinfer1::BuilderFlag::kFP16);



    // 4.构建网络结构，并赋值权重
    // 定义卷积核的权重
    float kernel_weight[] = {
        1, 0, 0,
        0, 1, 0,
        0, 0, 1
    };

    // 配置卷积参数
    nvinfer1::Weights conv1_weight;
    nvinfer1::Weights conv1_no_bias;
    conv1_weight.count = sizeof(kernel_weight) / sizeof(kernel_weight[0]);
    conv1_weight.type = nvinfer1::DataType::kFLOAT;
    conv1_weight.values = kernel_weight;
    conv1_no_bias.count = 0;
    conv1_no_bias.type = nvinfer1::DataType::kFLOAT;
    conv1_no_bias.values = nullptr;


    // 画个圈圈叫输入
    auto input = network->addInput(
        "Image",                                                // 指定输入节点的名称
        nvinfer1::DataType::kFLOAT,                             // 指定输入节点的数据类型
        nvinfer1::Dims4(1, 1, 3, 3)                             // 指定输入节点的shape大小
    );

    // 画个圈圈叫卷积
    auto conv1 = network->addConvolution(
        *input,                                                 // 输入节点的tensor，需要提供引用
        1,                                                      // 指定输出通道数
        nvinfer1::DimsHW(3, 3),                                 // 卷积核的大小
        conv1_weight,                                           // 卷积核参数
        conv1_no_bias                                           // 偏置参数，这里没有设置偏置
    );
    conv1->setName("conv1");                                    // 设置卷积名称
    conv1->setStride(nvinfer1::DimsHW(1, 1));                   // 设置卷积stride
    conv1->setPadding(nvinfer1::DimsHW(0, 0));                  // 设置卷积padding
    conv1->setDilation(nvinfer1::DimsHW(1, 1));                 // 设置卷积膨胀率

    // 画个圈圈叫激活，ReLU
    auto relu1 = network->addActivation(
        *conv1->getOutput(0),                                   // 指定其输入为卷积的输出
        nvinfer1::ActivationType::kRELU                         // 指定激活函数的类型为ReLU
    );
    relu1->setName("relu1");


    // 定义网络的输出节点
    auto output = relu1->getOutput(0);                          // 获取ReLU的输出
    output->setName("Predict");                                 // 设置输出节点的名称叫Predict
    network->markOutput(*output);                               // 告诉网路，这个节点是输出节点，值将会保留


    // 5.使用构建好的网络，编译为引擎
    auto engine = builder->buildEngineWithConfig(*network, *config);


    // 6.序列化模型为数据，并存储为文件
    auto host_memory = engine->serialize();
    save_to_file("../trtmodel/lesson_01_simple_cnn.plan", host_memory->data(), host_memory->size());

    std::cout << "Build done." << std::endl;


    host_memory->destroy();
    engine->destroy();
    config->destroy();
    network->destroy();
    builder->destroy();


}



// 模型推理
void model_inference()
{
    JLogger logger;
    cudaStream_t stream = nullptr;

    // 设置推理用的设备，创建流
    cudaSetDevice(0);
    cudaStreamCreate(&stream);

    // 2.加载模型数据
    auto model_data = load_from_file("../trtmodel/lesson_01_simple_cnn.plan");
    if(model_data.empty())
    {
        std::cout << "Load model failure." << std::endl;
        cudaStreamDestroy(stream);
        return;
    }

    // 3.创建运行时实例对象，并反序列化模型
    //   通过引擎，创建执行上下文
    auto runtime = nvinfer1::createInferRuntime(logger);
    auto engine = runtime->deserializeCudaEngine(model_data.data(), model_data.size());
    auto context = engine->createExecutionContext();

    // 4.获取绑定的tensor信息，并打印出来
    //   所谓绑定的tensor，就是指输入和输出节点
    int nbindings = engine->getNbBindings();

    // 打印绑定tensor的相关信息
    std::cout << "nbingdings = " << nbindings << std::endl;
    for(int i = 0; i < nbindings; ++i)
    {
        auto dims = engine->getBindingDimensions(i);
        auto name = engine->getBindingName(i);
        auto type = engine->bindingIsInput(i) ? "Input": "output";
        printf("Binging %d [%s], dimension is %s, type is %s.\n",
                                                            i,
                                                            name,
                                                            format_dim(dims).c_str(),
                                                            type);
    }

    // 5.准备输入数据和输出数据的内存空间
    float intput_data[] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };

    float* input_device = nullptr;
    float* output_device = nullptr;

    cudaMalloc(&input_device, sizeof(intput_data));
    cudaMalloc(&output_device, sizeof(float));

    cudaMemcpyAsync(input_device, intput_data, sizeof(intput_data), cudaMemcpyHostToDevice, stream);

    // 这里的bindings设备指针的顺序，必须与bindings的索引相对应
    void* bindings_device_pointer[] = {input_device, output_device};

    // 6.入队并进行推理
    bool finished = context->enqueueV2(bindings_device_pointer, stream, nullptr);
    if(!finished)
    {
        printf("Enqueue failure.\n");
    }


    // enqueueV2的最后一个参数是inputConsumed，是通知input_device可以被修改的事件指针
    // 如果在这里cudaEventSynchronize(inputConsumed);，在这句同步以后，input_device就可以被修改干别的事情
    // 这里使用input_device干别的事情


    // 收集执行结果
    float output_value = 0;
    cudaMemcpyAsync(&output_value, output_device, sizeof(float), cudaMemcpyDeviceToHost, stream);

    // 同步流，等待执行完成
    cudaStreamSynchronize(stream);



    // 7.打印最后结果
    printf("output_value = %f\n", output_value);


    // 释放自己分配的内存
    cudaFree(input_device);
    cudaFree(output_device);


    // 8.释放TensorRT分配的内存
    context->destroy();
    engine->destroy();
    runtime->destroy();
    cudaStreamDestroy(stream);

}






// 在main函数中调用
int main()
{
    // 编译
    // model_build();
    // printf("Build Done.\n");


    // 推理
    model_inference();
    printf("Inference Done.\n");


    return 0;
}













































