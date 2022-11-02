// #include <stdio.h>
// #include <vector>
// #include<builder/trt_builder.hpp>
// #include<infer/trt_infer.hpp>
// #include <opencv2/opencv.hpp>


// void model_build()
// {
//     TRT::compile(
//         TRT::Mode::FP16,
//         3,
//         TRT::ModelSource("../resnet18_classifier.onnx"),
//         TRT::CompileOutput("../trtmodel/resnet18_classifier_fp16.plan"),
//         {TRT::InputDims({1, 3, 224, 224})},
//         nullptr
//     );

//     printf("Build Done.\n");

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

//     auto engine = TRT::load_infer("../trtmodel/resnet18_classifier_fp16.plan");
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
//     printf("Using TRT framework.\n");

//     // 采用TRT框架对模型编译
//     // model_build();


//     // 采用TRT框架对模型进行推理
//     model_inference();




//     return 0;
// }













































