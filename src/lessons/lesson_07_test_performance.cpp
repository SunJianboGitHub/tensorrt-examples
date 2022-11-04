// #include <stdio.h>
// #include <time.h>
// #include <string>
// #include <opencv2/opencv.hpp>
// #include <common/ilogger.hpp>

// #include <builder/trt_builder.hpp>
// #include <infer/trt_infer.hpp>




// void model_build(int batch, const char* name)
// {
//     TRT::set_device(0);

//     // 如果有必要，请初始化他，因为你的网络如果引用了特定op，需要nv实现的插件支持
//     TRT::init_nv_plugins(); 

//     const std::string save_name = std::string("../trtmodel/resnet18_classifier_") + std::string(name) + std::string("_") + std::to_string(batch) + ".plan";

//     if(name == "fp32")
//     {
//         TRT::compile(
//             TRT::Mode::FP32,
//             batch,
//             TRT::ModelSource("../resnet18_classifier.onnx"),
//             TRT::CompileOutput(save_name),
//             {TRT::InputDims({-1, 3, 224, 224})},
//             nullptr
//         );
//     }
//     if(name == "fp16")
//     {
//         TRT::compile(
//             TRT::Mode::FP16,
//             batch,
//             TRT::ModelSource("../resnet18_classifier.onnx"),
//             TRT::CompileOutput(save_name),
//             {TRT::InputDims({-1, 3, 224, 224})},
//             nullptr
//         );
//     }
//     if(name == "int8")
//     {
//         auto int8process = [] (int current, int count, const std::vector<std::string>& images, std::shared_ptr<TRT::Tensor>& tensor){
//             printf("Calibrator %d / %d.\n", current, count);
//             for(int i = 0; i < images.size(); ++i){
//                 // int8 compilation requires calibration. We read image data and set_norm_mat. Then the data will be transfered into the tensor.
//                 auto image = cv::imread(images[i]);
//                 cv::resize(image, image, cv::Size(224, 224));
//                 float mean[] = {0.485, 0.456, 0.406};
//                 float std[] = {0.229, 0.224, 0.225};
//                 tensor->set_norm_mat(i, image, mean, std);
//             }
//         };

//         TRT::compile(
//             TRT::Mode::INT8,
//             batch,
//             TRT::ModelSource("../resnet18_classifier.onnx"),
//             TRT::CompileOutput(save_name),
//             {TRT::InputDims({-1, 3, 224, 224})},
//             int8process,
//             "../example_images/",
//             "../trtmodel/Calibrator.txt"
//         );
//     }

//     return;

// }








// // softmax 用于后处理
// void softmax_inplace(float* predict, int nums)
// {
//     float sum = 0;
//     for(int i = 0; i < nums; ++i)
//     {
//         sum += exp(predict[i]);
//     }
//     for(int i = 0; i < nums; ++i)
//     {
//         predict[i] = exp(predict[i]) / sum;
//     }
// }



// // 模型推理
// void model_inference(int batch, int num_images, const char* name)
// {

//     TRT::set_device(0);

    
//     auto model = TRT::load_infer(iLogger::format("../trtmodel/resnet18_classifier_%s_%d.plan", name, batch));
//     if(model == nullptr)
//     {
//         printf("Load engine failure.\n");
//         return;
//     }
    
//     auto input = model->input(0);
//     auto output = model->output(0);
//     input->resize_single_dim(0, batch);                         // 切记这里不要使用resize(batch)，否则会报错

//     cv::Mat image = cv::imread("../example_images/dog_2.jpg");
//     cv::resize(image, image, cv::Size(224, 224));
//     cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

//     float mean[] = {0.485, 0.456, 0.406};
//     float std[] = {0.229, 0.224, 0.225};
//     for(int i = 0; i < batch; ++i)
//     {
//         input->set_norm_mat(i, image, mean, std);
//     }

//     // 开始计时
//     time_t start;
//     start = clock();

//     // 多批次的前向推理
//     int niters = ceil(num_images / (float)batch);
//     for(int i = 0; i < niters; ++i)
//     {
//         model->forward();
//     }
    
//     float total_times = ((float)(clock() - start) / CLOCKS_PER_SEC) * 1000;         // 单位是毫秒

//     // 后处理一个检测结果
//     float* predict = output->cpu<float>(0);
//     int predict_channel = output->channel();
//     softmax_inplace(predict, predict_channel);
//     int label = std::max_element(predict, predict + predict_channel) - predict;
//     float confidence = predict[label];

//     printf("\n\n%d images. Inference %s[%d batch] times = %.3f ms, %.3f ms / image, Predict label = %d, confidence = %f.\n\n\n",
//             num_images, name, batch, total_times, total_times/num_images, label, confidence);

//     return;
// }



// // 编译单精度、半精度、INT8模型
// void test_build(int batch)
// {
//     model_build(batch, "fp32");
//     model_build(batch, "fp16");
//     model_build(batch, "int8");
//     printf("Build Done.\n\n");
// }



// // 针对单精度、半精度以及int8模型进行推理，获取它们的速度
// void test_performance(int batch, int num_images)
// {
//     num_images = ceil(num_images / (float)batch) * batch;           // 计算出一个batch的整数倍
//     model_inference(batch, num_images, "int8");                     // int8推理
//     model_inference(batch, num_images, "fp16");                     // fp16推理
//     model_inference(batch, num_images, "fp32");                     // fp32推理
    
// }







// int main()
// {
//     int batch[] = {1, 4, 8, 16, 32};

//     // 编译阶段
//     for(int i = 0; i < sizeof(batch) / sizeof(batch[0]); ++i)
//     {
//         int b = batch[i];
//         test_build(b);
//     }


//     // 推理阶段
//     for(int i = 0; i < sizeof(batch) / sizeof(batch[0]); ++i)
//     {
//         int b = batch[i];
//         test_performance(b, 1000);
//         // break;
//     }

//     printf("Test Done.\n");

//     return 0;
// }





































