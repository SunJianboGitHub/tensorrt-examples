#include <stdio.h>
#include <cstring>

#include <builder/trt_builder.hpp>
#include <infer/trt_infer.hpp>







void model_build()
{
    TRT::compile(
        TRT::Mode::FP32,
        1,
        TRT::ModelSource("../plugin_SwishAndBias.onnx"),
        TRT::CompileOutput("../trtmodel/plugin_SwishAndBias.plan"),
        {},
        nullptr
    );
}



void model_inference()
{

    TRT::set_device(0);
    auto model = TRT::load_infer("../trtmodel/plugin_SwishAndBias.plan");

    auto input = model->input(0);
    auto output = model->output(0);
    float input_data[] = {
        0, 1, 2,
        3, 4, 5,
        6, 7, 8
    };

    memcpy(input->cpu<float>(), input_data, sizeof(input_data));

    model->forward();

    float* predict = output->cpu<float>();

    printf("Predict is %f.\n", *predict);

}




int main()
{
    // 模型编译
    // model_build();

    // 模型推理
    model_inference();

    return 0;
}





























