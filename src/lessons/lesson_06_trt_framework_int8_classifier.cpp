// #include <stdio.h>
// #include <memory>
// #include <builder/trt_builder.hpp>
// #include <infer/trt_infer.hpp>
// #include <opencv2/opencv.hpp>


// /*
//  * 常见的量化方式：
//  *      1. 训练后量化(PTQ, Post Training Quantization)
//  *      2. 量化感知训练(QAT, Quantization Aware Training)
//  * 
//  * 方法：
//  *      1. PTQ 就是pytorch导出ONNX模型，然后采用TensorRT的工具转换为INT8格式的engine，当存在不支持算子时，需要手写实现
//  *      2. QAT 就是在pytorch中训练模型时，直接采用NVIDIA提供的工具包，直接训练INT8格式的模型，导出为ONNX模型，然后再通过TensorRT加速
//  * 
//  * 优势与劣势：
//  *      1. 模型量化为INT8之后，模型的推理速度提高，至少提升一倍
//  *      2. 速度的提高，带来了精度的降低，大约在-10% ~ 1%， 平均也在-5%左右
//  *      3. 要评估业务能不能容忍精度的降低，不过有时候还可能提高精度。
//  *      4. 前提是硬件支持INT8推理，否则影响速度
//  *      5. INT8量化在检测模型上精度会显著下降，因此我们普遍采用fp16进行检测推理
// */



// /*
//  * TensorRT中的PTQ量化细节：
//  *      1. 首先是把参数转化为INT8格式，使用INT8乘法来代替fp32乘法，从而实现加速，因此存在精度损失
//  *      2. 量化之前的情况
//  *          a. 输入参数 x[fp32], 权重参数 weight[fp32], 偏置参数 bias[fp32]
//  *          b. 输出结果为 y = x[fp32] * weight[fp32] + bias[fp32]
//  *      3. 量化之后的情况
//  *          a. 输入参数转化为INT8，x_int8 = int8(x), 这一步也称之为编码,编码参数很重要，因此后续会有校准环节
//  *          b. 再把权重参数转化为INT8， weight_int8 = int8(weight)
//  *          c. INT8乘法，c = (x_int8 * weight_int8) 得到的是int16的结果，这一步由硬件实现
//  *          d. c -> c[fp32], 这一步称之为解码，通过之前的编码参数来解码
//  *          e. 最终的输出 y = c[fp32] + bias[fp32], 数据类型是fp32
//  * 
//  * TensorRT中的标定：
//  *      1. 寻找并确定编码参数
//  *      2. 其中一种算法是，找到low、high并确定为缩放区间，使用KL散度来计算区间
//  *          a. 转化为INT8权重后，与转化前的分布要尽可能相近
//  *          b. 衡量两个分布之间的差异，用KL散度，也叫做相对熵
//  *      3. 输入数据，使用数据进行推理，确定合适的编码参数，并采用该参数进行量化
//  *      4. 尽量采用任务类似的数据，数据多一些，标定结果会好一点
//  *      5. 标定是一个缓慢的过程，特别是在嵌入式设备上
//  * 
// */

// // TRT::Int8Process
// auto int8preocess = [](int current, int count, const std::vector<std::string>& files, std::shared_ptr<TRT::Tensor>& tensor){
//     for(int i = 0; i < files.size(); ++i)
//     {
//         auto image = cv::imread(files[i]);
//         cv::resize(image, image, cv::Size(224, 224));
//         float mean[] = {0.485, 0.456, 0.406};                          
//         float std[] = {0.229, 0.224, 0.225};
//         tensor->set_norm_mat(i, image, mean, std);

//         printf("Calibrate %d / %d.\n", current, count);

//     }
// };





// void model_build()
// {
//     TRT::set_device(0);                 // 设置设备索引，确定在那个设备上进行编译

//     TRT::compile(
//         TRT::Mode::INT8,
//         1,
//         TRT::ModelSource("../resnet18_classifier.onnx"),
//         TRT::CompileOutput("../trtmodel/resnet18_classifier_int8.plan"),
//         {TRT::InputDims({-1, 3, 224, 224})},
//         int8preocess,
//         "../example_images/",
//         "../trtmodel/calibrator.txt"
//     );


// }



// // softmax后处理
// void softmax_inplace(float* predict, int n)
// {
//     float sum = 0;
//     for(int i = 0; i < n; ++i)
//     {
//         sum += exp(predict[i]);
//     }
//     for(int i = 0; i < n; ++i)
//     {
//         predict[i] = exp(predict[i]) / sum;
//     }
// }


// void model_inference()
// {
//     TRT::set_device(0);

//     /* 这是一个很好的设计思想,火情识别中也采用这种思想
//      * 
//      * RAII (Resource Acquisition Is Initialization) 资源获取即初始化
//      * 资源创建的时候，同时把初始化做了，比如 Person* person = new Person("alex");这就是所谓的RAII
//      * 再结合哪里分配哪里释放这个概念，就形成了这里提到的RAII
//      * 再配合接口模式，就形成了我们的load_infer函数,
//      * 
//      * 
//     */

//     auto engine = TRT::load_infer("../trtmodel/resnet18_classifier_int8.plan");
//     auto input = engine->input(0);
//     auto output = engine->output(0);

//     cv::Mat image = cv::imread("../example_images/dog_2.jpg");    // 读取图像
//     cv::resize(image, image, cv::Size(224, 224));                   // 缩放图像到224*224
//     cv::cvtColor(image, image, cv::COLOR_BGR2RGB);                  // 将BGR转换为RGB

//     float mean[] = {0.485, 0.456, 0.406};                           // ImageNet上的均值，对应RGB通道
//     float std[] = {0.229, 0.224, 0.225};                            // ImageNet上的方差，对应RGB通道

//     input->set_norm_mat(0, image, mean, std);
//     engine->forward();

//     float* predict = output->cpu<float>();                          // 得到推理结果
//     int predict_channel = output->channel();                        // 得到输出的维度

//     softmax_inplace(predict, predict_channel);                      // 进行softmax处理

//     int label = std::max_element(predict, predict + predict_channel) - predict;         // 获取最大值索引，也就是类别ID
//     float confidence = predict[label];                                                  // 获得类别对应的概率

//     printf("label is %d, confidence is %f.\n", label, confidence);

//     printf("Inference Done.\n");
// }





// int main()
// {

//     // INT8 模型编译
//     // model_build();

//     // INT8模型推理
//     model_inference();

//     printf("Quantization INT8 use trt framework.\n");
//     return 0;
// }












































