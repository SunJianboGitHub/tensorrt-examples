// #include <stdio.h>
// #include <memory>
// #include <string.h>


// /*
//  * 不要试图返回一个栈内存上的指针，因为函数结束，内存就会释放，会导致访问错误
//  * AA函数是合法的函数
//  * BB函数是不合法的，会造成野指针
//  * 
//  * 
//  * 返回这种 c style 指针，只有一种场合(合适的做法)
//  *      1. 类的函数可以返回 c style 指针，还要保证指针不会被释放，是类成员
//  *      2. 返回常量区的指针
//  * 
//  * 
//  * 内存分配的原则性问题:
//  *      1. 哪里分配哪里释放，因为外面的人没办法知道如何合理析构
//  *          - 比如 malloc分配的内存需要free
//  *          - new分配的内存要delete
//  *          - new obj[]分配的内存要delete [] obj
//  *          - TensorRT的对象，要obj->destroy()
//  * 
//  * 解决内存分配释放问题
//  *      1. 使用类实例，利用类来管理分配和释放，自动进行
//  *      2. shared_ptr, 共享指针，也叫智能指针，它具备自动释放的功能，C++11加入的类
//  * 
//  * 
// */

// char* AA()
// {
//     char* name = "hello";                           // 这里name指针指向常量区，是可以作为返回值的

//     return name;
// }



// // 下面的代码是有问题的，会有野指针
// // char* BB()
// // {
// //     char name[] = "hello";                      // 虽然hello在常量区，但是name数组却在栈内存中，不可以作为返回值

// //     // 上面的语句其实等价于下面的写法
// //     // char name[6];
// //     // strcpy(name, "hello");

// //     return name;                                // 返回之后将成为野指针
// // }


// // 与智能指针匹配的释放函数
// void free_cc(char* ptr)
// {
//     printf("Free cc %p.\n", ptr);

//     if(ptr != nullptr)
//     {
//         free(ptr);
//     }
// }




// std::shared_ptr<char> CC()
// {
//     char* name = (char*)malloc(6);                      // 虽然这里申请了堆内存，但是采用了智能指针管理，因此不会出现内存泄漏

//     // 如果中间出错，其实还是会存在内存泄漏，采用RAII+智能指针才能做好
//     printf("Malloc cc %p.\n", name);

//     strcpy(name, "hello");


//     return std::shared_ptr<char>(name, free_cc);
// }








// int main()
// {

//     auto cc = CC();
//     printf("cc is %s.\n", cc.get());


//     return 0;
// }









































