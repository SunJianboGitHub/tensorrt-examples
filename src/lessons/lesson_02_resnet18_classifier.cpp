// #include <stdio.h>
// #include <iostream>
// #include <NvInfer.h>
// #include <onnx_parser/NvOnnxParser.h>
// #include <cuda_runtime.h>
// #include <opencv2/opencv.hpp>
// #include <vector>

// #include "filetools.hpp"


// // 1.定义Logger类
// //     TensorRT内部出现的任何消息，都会通过Logger类打印出来
// //     根据消息等级区分问题，有助于排查bug
// // 纯虚函数
// //     a.具有纯虚函数的类，不能够被实例化
// //     b.纯虚函数是只有声明没有实现的虚函数，也就是尾巴后面给 =0
// //     c.作用是，声明函数，实现交给子类，也就是常说的接口类
// //     d.如果子类继承的基类中存在纯虚函数，如果需要实例化子类，则必须实现纯虚函数
// class JLogger: public nvinfer1::ILogger
// {
//     public:
//         virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override
//         {
//             printf("LOG[%d]: %s\n", severity, msg);
//         }
// };




// // 模型编译
// void model_build()
// {
//     JLogger logger;

//     // 设置显示batch
//     u_int32_t flag = 1U << static_cast<u_int32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

//     nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
//     nvinfer1::INetworkDefinition* network = builder->createNetworkV2(flag);
//     nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();

//     nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
//     profile->setDimensions("Image", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, 3,224, 224));
//     profile->setDimensions("Image", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, 3,224, 224));
//     profile->setDimensions("Image", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(1, 3,224, 224));

    
//     builder->setMaxBatchSize(3);
//     config->setMaxWorkspaceSize(1 << 30);
//     // config->setFlag(nvinfer1::BuilderFlag::kFP16);          // 半精度推理，注释掉，默认是单精度推理fp32
//     config->addOptimizationProfile(profile);


//     // 构建网络，这里使用的是ONNX
//     nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
//     if(!parser->parseFromFile("../resnet18_classifier.onnx", 1))
//     {
//         printf("Parse failed.\n");

//         parser->destroy();
//         config->destroy();
//         network->destroy();
//         builder->destroy();

//         return;
//     }



//     nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);                                  // 根据配置，编译为engine
//     nvinfer1::IHostMemory* host_memory = engine->serialize();                                                           // engine序列化到内存

//     bool finished = save_to_file("../trtmodel/resnet18_classifier_fp32.plan", host_memory->data(), host_memory->size());

//     printf("Build done.\n");



//     // 释放tensorRT申请的资源
//     host_memory->destroy();
//     engine->destroy();

//     parser->destroy();
//     config->destroy();
//     network->destroy();
//     builder->destroy();

// }


// // 模型推理
// void model_inference()
// {
//     JLogger logger;

//     auto model_data = load_from_file("../trtmodel/resnet18_classifier_fp32.plan");
//     if(model_data.empty())
//     {
//         printf("Load model failure.\n");
//     }

//     nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
//     nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(model_data.data(), model_data.size());
//     nvinfer1::IExecutionContext* context = engine->createExecutionContext();

//     // 推理过程
//     cv::Mat image = cv::imread("../example_images/dog_2.jpg");                        // 读取图像
//     cv::resize(image, image, cv::Size(224, 224));                   // 缩放图像到224*224
//     cv::cvtColor(image, image, cv::COLOR_BGR2RGB);                  // 将BGR转换为RGB

//     cv::Scalar mean(0.485, 0.456, 0.406);                           // ImageNet上的均值，对应RGB通道
//     cv::Scalar std(0.229, 0.224, 0.225);                            // ImageNet上的方差，对应RGB通道

//     cv::Mat image_float;
//     // 转换image到浮点数，image_float = image * alpha + beta;
//     image.convertTo(image_float, CV_32F, 1 / 255.0f);               // 归一化图像到float类型
//     image_float = (image_float - mean) / std;                       // 标准化

//     // OpenCV 读取的image的像素排列是 rgb rgb rgb
//     // 模型推理需要的输入维度是 NCHW，也就是 rrrrr ggggg bbbbb
//     // 因此，需要对OpenCV读取的图像变换
//     // 也可以采用像素遍历的方式，先遍历行，因为是行排列优先
//     std::vector<float> input_host_image(image_float.rows * image_float.cols * image_float.channels());


//     // 让下面3个通道，分别引用input_host_image的地址
//     cv::Mat channel_base_reference[3];
//     float* input_host_image_ptr = input_host_image.data();
//     for(int i = 0; i < 3; ++i)
//     {
//         channel_base_reference[i] = cv::Mat(image_float.rows, image_float.cols, CV_32F, input_host_image_ptr);
//         input_host_image_ptr += image_float.rows * image_float.cols;
//     }

//     // 把image_float根据通道拆解，分别放在3个Mat中
//     // split必须确保image_float与channel_base_reference提供的大小类型通道一致，
//     // 否则，会导致channel_base_reference内存被重新分配，也就是断开了引用关系，导致结果与预期不符合
//     cv::split(image_float, channel_base_reference);

//     // 如果不小心造成了channel_base_reference释放了引用，分配了新空间，将会导致那一发现的错误
//     // 这里做一个断言，如果有异常，提示出来
//     CV_Assert((void*)channel_base_reference[0].data == (void*)input_host_image.data());


//     // 开始准备转GPU推理
//     float* input_device_image = nullptr;
//     size_t input_device_image_bytes = input_host_image.size() * sizeof(float);
//     cudaMalloc(&input_device_image, input_device_image_bytes);

//     // 创建 cuda stream
//     cudaStream_t stream;
//     cudaStreamCreate(&stream);

//     // 异步拷贝数据
//     cudaMemcpyAsync(input_device_image, input_host_image.data(), input_device_image_bytes, cudaMemcpyHostToDevice, stream);


//     // 分配输出的设备空间
//     float* output_device = nullptr;
//     size_t output_device_bytes = 1000 * sizeof(float);
//     cudaMalloc(&output_device, output_device_bytes);

//     // 绑定输入输出，入队进行推理
//     void* bindings[] = {input_device_image, output_device};
//     context-> enqueue(1, bindings, stream, nullptr);

//     // 异步复制结果到 vector
//     std::vector<float> output_predict(1000);
//     cudaMemcpyAsync(output_predict.data(), output_device, output_device_bytes, cudaMemcpyDeviceToHost, stream);

//     // 同步流，确保结果拷贝完成
//     cudaStreamSynchronize(stream);


//     // 后处理，softmax操作转换为概率
//     float sum = 0;
//     for(float& item: output_predict)
//     {
//         sum += exp(item);
//     }
//     for(float& item: output_predict)
//     {
//         item = exp(item) / sum;
//     }


//     // 找出概率最大值的索引，确定标签
//     int label = std::max_element(output_predict.begin(), output_predict.end()) - output_predict.begin();
//     float confidence = output_predict[label];
//     printf("Predict label is %d, confidence is %.5f.\n", label, confidence);


//     cudaStreamDestroy(stream);
//     cudaFree(output_device);
//     cudaFree(input_device_image);


//     context->destroy();
//     engine->destroy();
//     runtime->destroy();
    


// }







// int main()
// {
//     printf("Hello TensorRT ResNet18 classification.\n");

//     // 模型编译
//     // model_build();

//     // 模型推理
//     model_inference();



//     return 0;
// }
































