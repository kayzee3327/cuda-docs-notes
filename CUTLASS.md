# CUTLASS 2.x layout

## `RowMajor`和`ColumnMajor`

在row-major储存中，矩阵中每一行的内存都连续，column-major同理。



# `TensorRef`

包含

- 起始地址的指针
- 用于访问的layout（注意是CUTLASS 2.x的概念，而非CuTe layout，此layout是跳转访问的方式，只负责将坐标作简单运算，不考虑边界）

# `TensorView`

在`TensorRef`的基础上增加了`extent`，使得访问有边界





# `HostTensor`

CUTLASS 2.x的layout与CuTe中提到的m-major，k-major，n-major逻辑不同。对比如下：

- m-major：对应row-major的`HostTensor({K, M})`或column-major的`HostTensor({M, K})`
