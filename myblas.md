# C API

## extern "C"和C++实现

`extern "C"`包括的代码在编译期间会以C风格链接，避免Name Mangling

### exceptions

在c++的编程模型中，出现exception会进行stack unwinding来找到对应的处理过程，而C中没有exception或stack unwinding的定义。尽管C++和C的stack frame结构相同，C编译器

- 没有自动销毁对象的过程（call deconstructor），只会抛弃local variable的内存区
- 没有throw/catch
- 没有类成员函数的隐藏this指针
- 等等

所以如果C接收到了C++的exception，由于内存分布不同，很有可能会访问错误的内存，引起segfault或UB

正确处理方法：

- 在可能抛出exception的调用加上`std::nothrow`
- 使用try-catch，可能需要定义宏来方便编写

### `std::nothrow`

它告诉编译器启用function overload resolution，比如例子

```cpp
myblas_context* ctx = new (std::nothrow) myblas_context{}; 
```

编译器会在`new`尝试分配堆上内存失败后使其不抛出异常，而是返回`nullptr`，但注意这不能保证`myblas_context`的构造不抛出异常

## API可见性

在编译时加入`-fvisibility=hidden`使得未被`__attribute__((visibility("default")))`标注的API或变量暴露，例如CUTLASS的API等，避免命名冲突和过大的编译文件

在cmake中，传入以下标签

```cmake
set_target_properties(myblas PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    OUTPUT_NAME "myblas"
    VERSION 0.1.0
    SOVERSION 0
    # Hide all symbols by default; only MYBLAS_API functions are exported.
    # CXX_VISIBILITY_PRESET covers host code; CUDA_VISIBILITY_PRESET covers
    # .cu files (nvcc passes -Xcompiler=-fvisibility=hidden to the host compiler).
    CXX_VISIBILITY_PRESET  hidden
    CUDA_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)
```

- `CXX_VISIBILITY_PRESET hidden`：给host compiler传入`-fvisibility=hidden`（编译cpp文件）
- ` CUDA_VISIBILITY_PRESET hidden`：给nvcc传入`-Xcompiler=-fvisibility=hidden`（编译cu文件）
- `VISIBILITY_INLINES_HIDDEN ON`：隐藏inline function的symbol



# cublas对比CuTe：Mindset Diff

尽管接口很相似，有transpose考虑，有lda/ldb/ldc，但二者看待矩阵形状和内存分布的视角完全不同。

## cublas（或BLAS）思想 - 数学

cublas是为了实现数学公式$C = \text{op}(A) \times \text{op}(B)$加速设计的，因此它更倾向于直接使用公式中的概念。

- Matrix A is logically $M \times K$.
- Matrix B is logically $K \times N$.
- Matrix C is logically $M \times N$.

例如leading dimension、row/column-major的设计都是为了按照逻辑形状来访问，它始终假设A应该以M为一行访问，lda是访问一行的下一个entry需要跳过的地址。这实际上是CuTe想要改进的shape-stride模式，才有了hierarchical layout

## CuTe思想 - 硬件

CuTe则想要一种最容易利用硬件的编程范式

- Matrix A is logically $M \times K$.

- Matrix B is logically **$N \times K$**.

- Matrix C is logically $M \times N$.

可以看到例子中都是以bM, bK和bN, bK作为smem shape的，这样的设计对向量化友好





# GEMM计算

## 如果alpha beta与ABC不同类型





# GEMM Traits设计

## 最初：分离configuration和execution

从一开始借鉴FA和CUTLASS，既然CuTe GEMM里利用了大量的编译期计算，比如TiledCopy、TiledMMA，那么每种GEMM需要的参数配置都可以归纳到一个traits里，初始设计如下：

- ABC type
- tile shape
- pipeline stages
- smem layouts
- gmem->smem TiledCopy
- smem->rmem Copy_Atom
- TiledMMA
- constexpr ThreadsPerBlock



## TiledCopy特化

处理NN/TN/NT/TT四种GEMM时，相同条件下参数都很相似，但是gmem->smem的拷贝需要不同的TV layouts，此时为每种GEMM写一个完整的Traits将产生冗长且不必要的代码，故创建`SelectGmemTiledCopy`类。这样编译器可以根据模板特化来生成4种GEMM，在运行时有选择地调用。

新的问题出现了，`SelectGmemTiledCopy`需要什么模板参数来精确地选择不同架构、不同transpose、不同problem shape的TiledCopy呢？

- 根据不同transpose设计，我们只需要调整TV layouts的Stride
- 根据不同架构、不同problem shape设计，则需要选择Copy_Atom、TV layouts的shape和stride

好在我们有一个重要的切入点：向量化访存，即所有拷贝都要尽可能打满128bit的向量化带宽

那么我们便可以根据tile shape和transpose，找到contiguous的mode，将其除以128bit



# GEMM Params设计

## 使用静态Stride

因为transpose是运行时才可知的，而stride需要根据N/T选择，设计成运行时判断stride很符合直觉，却不能利用CuTe的静态特性。而考虑直接在`set_params()`使用`auto dA = make_stride(...)`时，静态的Stride的类型需要提前告知`GemmParams`，这在逻辑上形成了一个悖论。

所以作为静态
