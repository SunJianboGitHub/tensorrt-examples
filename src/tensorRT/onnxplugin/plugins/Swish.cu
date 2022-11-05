#include <stdio.h>
#include <cuda_runtime.h>
#include <onnxplugin/onnxplugin.hpp>
#include <common/cuda_tools.hpp>


/*
 * 在hpp文件中，尽可能减少使用 using namespace 
 * 尤其是公共库的namespace不要写，坚决不要写
 * 头文件，要求使用什么就依赖什么，越少越简单越好
 * 否则，会造成复杂度高，难以管理，循环依赖等问题
*/

__device__ float sigmoid(float x)
{
    return 1 / (1 + exp(-x));
}



__global__ void swish_kernel_impl(float* input, float* output, float* param, int edge)
{
    KernelPositionBlock;
    float x = input[position];
    output[position] = x * sigmoid(x) + *param;
}




// 实现自己的定制化config类，如果不是特殊需求，一般不需要
class SwishLayerConfig: public ONNXPlugin::LayerConfig
{
    public:
        virtual void init() override;
};


// 实现干活的类, 集成自TRTPlugin
class Swish: public ONNXPlugin::TRTPlugin
{
    public:
        // 设置这个插件，使用宏指令
        // 同时告诉插件，我的名字叫做Swish
        SetupPlugin(Swish);

        virtual void config_finish() override
        {
            printf("info is: %s.\n", config_->info_.c_str());
        }

        // 重写config函数，返回自定义的LayerConfig实例
        virtual std::shared_ptr<ONNXPlugin::LayerConfig> new_config() override
        {
            auto cfg = ONNXPlugin::TRTPlugin::new_config();
            //cfg->support_dtype_set_ = {nvinfer1::DataType::kHALF, nvinfer1::DataType::kFLOAT};
            cfg->support_dtype_set_ = {nvinfer1::DataType::kFLOAT};
            return cfg;
            
        }

        // 重写输出维度函数，返回这个layer所需要输出的shape
        // 这个函数是，如果头N个输出，则调用N次
        virtual nvinfer1::DimsExprs getOutputDimensions(int32_t outputIndex, const nvinfer1::DimsExprs* inputs, int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept override
        {
            return inputs[0];
        }

        // 重写，执行过程，如果返回值为0，表示没问题，否则有问题
        int enqueue(const std::vector<ONNXPlugin::GTensor>& inputs,
                    std::vector<ONNXPlugin::GTensor>& outputs, 
                    const std::vector<ONNXPlugin::GTensor>& weights, 
                    void* workspace, 
                    cudaStream_t stream) override
        {
            // 计算的就是所有维度的乘积
            // 这个插件只有一个输入就行，所以这里取input[0]
            int jobs = inputs[0].count();
            auto blocks = CUDATools::grid_dims(jobs);           // 向上取整
            auto threads = CUDATools::block_dims(jobs);

            swish_kernel_impl<<<blocks, threads, 0, stream>>>(
                inputs[0].ptr<float>(),
                outputs[0].ptr<float>(),
                weights[0].ptr<float>(),
                jobs
            );

        }

};



// 注册这个layer，让系统可以识别
// 这个宏里面实现了对工厂进行添加创建器的操作，使得系统在解析阶段时，builtin_op_importers.cpp中可以获取到，并认识
RegisterPlugin(Swish);




























