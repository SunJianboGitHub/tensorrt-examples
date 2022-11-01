#include <string>
#include <stdio.h>
#include <vector>
#include <NvInfer.h>




bool save_to_file(const std::string& file_name, const void* data, size_t data_size)
{

    FILE* handle = fopen(file_name.c_str(), "wb");
    if(handle == nullptr)
    {
        return false;
    }

    fwrite(data, 1, data_size, handle);
    fclose(handle);

    return true;
}


std::vector<unsigned char> load_from_file(const std::string& file_name)
{
    std::vector<unsigned char> output;

    FILE* handle = fopen(file_name.c_str(), "rb");
    if(handle == nullptr)
    {
        return output;
    }

    // 1.SEEK_END，移动文件指针到相对末尾位置偏移 0 个字节，第二个参数表示相对偏移量
    fseek(handle, 0, SEEK_END);

    // 2.获取当前文件指针位置，得到大小
    long size = ftell(handle);

    // 3.恢复文件指针到文件开头
    // 直接设置文件指针到特定位置，这里是0，就是文件开头了
    fseek(handle, 0, SEEK_SET);

    // 4.如果文件长度大于0，才有必要读取和分配
    if(size > 0)
    {
        output.resize(size);
        fread(output.data(), 1, size, handle);
    }

    // 5.关闭文件句柄，返回结果
    fclose(handle);
    
    return output;
}



const std::string format_dim(const nvinfer1::Dims& dims)
{
    char buffer[100] = {0};
    char *p = buffer;
    for(int i = 0; i < dims.nbDims; ++i)
    {
        const char* format = i == 0 ? "%d": " x %d";
        int string_length = sprintf(p, format, dims.d[i]);
        p += string_length;
    }

    return buffer;
}



























