
source_dir := src
object_dir := objs



# 查找cpp文件
cpp_srcs := $(shell find $(source_dir) -name "*.cpp")
cpp_objs := $(patsubst $(source_dir)/%.cpp, $(object_dir)/%.o, $(cpp_srcs))

# 查找cuda文件
# 由于cpp可能与cu文件同名，但是是不同文件，因此对于cuda程序，把cu改成cuo
cu_srcs := $(shell find $(source_dir) -name "*.cu")
cu_objs := $(patsubst $(source_dir)/%.cu, $(object_dir)/%.cuo, $(cu_srcs))


# 定义名称参数
workspace := bin
binary := pro

# 这里定义头文件、库文件和链接目标没有加-I -L -l，后面用foreach一次性增加
include_paths := /datav/lean/protobuf3.11.4/include \
				 /datav/lean/cuda-11.2/include \
				 /datav/lean/cudnn8.2.2.26/include \
				 /datav/lean/opencv-4.2.0/include/opencv4 \
				 /datav/lean/TensorRT-8.0.3.4.cuda11.3.cudnn8.2/include \
				 src \
				 src/tensorRT \
                 src/tensorRT/onnx \
				 src/tensorRT/common
				 
# 这里需要清除认识链接的库到底链接的是谁，这个非常重要
# 需要链接的对象，一定要是你预期的链接库
library_paths := /datav/lean/protobuf3.11.4/lib \
				 /datav/lean/cuda-11.2/lib64 \
				 /datav/lean/cudnn8.2.2.26/lib \
				 /datav/lean/opencv-4.2.0/lib \
				 /datav/lean/TensorRT-8.0.3.4.cuda11.3.cudnn8.2/lib \

				 


link_librarys := opencv_core opencv_imgcodecs opencv_imgproc \
				 cuda cudnn cudart cublas \
				 nvinfer nvinfer_plugin \
				 protobuf  pthread stdc++

# 定义编译选项，-w屏蔽警告
cpp_compile_flags := -m64 -fPIC -g -O0 -std=c++11 -w -fopenmp
cu_compile_flags  := -m64 -Xcompiler -fPIC -g -O0 -std=c++11 -w -Xcompiler -fopenmp -gencode=arch=compute_70,code=sm_70


# 对头文件、库文件、目标统一增加 -I -L -l
# -I 指定编译时头文件查找路径
# -L 指定链接目标时查找的目录
# -l 指定链接的目标名称，符合libname.so，-lname 规则
# rpaths,指定运行时路径
rpaths		  := $(foreach item,$(library_paths),-Wl,-rpath=$(item))
include_paths := $(foreach item,$(include_paths),-I$(item))
library_paths := $(foreach item,$(library_paths),-L$(item))
link_librarys := $(foreach item,$(link_librarys),-l$(item))


# 合并选项
cpp_compile_flags += $(include_paths)
cu_compile_flags  += $(include_paths)
link_flags        := $(rpaths) $(library_paths) $(link_librarys)


# 定义cpp的编译方式
$(object_dir)/%.o : $(source_dir)/%.cpp
	@mkdir -p $(dir $@)
	@echo Compile $<
	@g++ -c $< -o $@ $(cpp_compile_flags)


# 定义cuda文件的编译方式
$(object_dir)/%.cuo : $(source_dir)/%.cu
	@mkdir -p $(dir $@)
	@echo Compile $<
	@nvcc -c $< -o $@ $(cu_compile_flags)



# 定义workspace/pro文件的链接
$(workspace)/$(binary) : $(cpp_objs) $(cu_objs)
	@mkdir -p $(dir $@)
	@echo Link $@
	@g++ $^ -o $@ $(link_flags)


# 定义pro的快捷编译指令，这里只发生编译，不执行
pro : $(workspace)/$(binary)



# 定义编译并执行的指令，并且执行目录切换到workspace下
run : pro
	@cd $(workspace) && ./$(binary)


# 测试cuda程序执行效率，并产生分析文件
test_cuda_time : pro
	@cd $(workspace) && nvprof -o demo.nvvp -s ./$<



debug :
	@echo $(cpp_objs)
	@echo $(cu_objs)



clean :
	@rm -rf $(object_dir) $(workspace)/$(binary)


# 指定伪标签，作为指令
.PHONY : clean debug run pro














