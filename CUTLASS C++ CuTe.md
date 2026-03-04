# Layout

CuTe Layout由两部分组成

- Shape: 把任意input coordinate转化为natural coordinate，取模
- Stride: 把任意natural coordinate转化为内存中的index，做内积

## 怎么读

- 要记得，程序员通过layout找到内存中的一些数据，可能连续可能间断，把他们放进一个矩阵中统一做操作
- hierarchical coordinate不是目标，是出发点，因为把坐标拆分成了几个层次，才能有几个层次的stride，在内存中跳来跳去找到元素
- 所以hierarchical coordinate也没有增加矩阵的维度，无需通过维度来理解，逻辑上的维度一般只有最外层的坐标数量
- stride是每一维上增加到下一坐标的跳跃间隔
- 坐标从左向右按模运算增长 (colexicographical)，与数学中`(x,y,z,w,...)`的顺序相同
  - 比如shape(4, (4, 2))，这是一个2D矩阵，row为4，col上的坐标从0开始是`(0,0)` `(1,0)` `(2,0)` `...`；总坐标如此增长`(0,(0,0))` `(1,(0,0))` `(2,(0,0))`  `(3,(0,0))` `(0,(1,0))` `...`

## 什么是compatible shape/layout



## 怎么将各个compatible shape中的coordinate相互转化

1. 从左向右对shape取模
2. 如果有更深层次的，对

## 怎么将coordinate转化到实际内存位置

# Layout Algebra

## Coalesce

输入一个复杂的Layout，输出简化的Layout，前后Layout对于相同的coordinate，映射的物理内存位置一样

例子：`Layout (2,(1,6)):(1,(6,2))`经过coalesce操作后转化为`Layout 12:1`

转化方法见docs

## By-mode Coalesce

在Coalesce时，传入一个参数代表着期望维度，这里侧面印证了hierarchical layout不影响维度

例子：`Layout (2,(1,6)):(1,(6,2))`经过`coalesce(a, Step<_1,_1>{})`操作后转化为`Layout (2,6):(1,2)`

## Composition

### 性质

记layout `A` `B` composition后layout为`R`

- `R(c) := (A o B)(c)= A(B(c))`
- `R`与`B` compatible

### 计算

根据observations，不失一般性地

- 假设B是整型layout，因为最终可以拆分为整型的sublayout，
- 假设A是flattened，因为在上述composition计算中，传入A的是1D coordinate
- 假设A是coalesced，因为假设A拥有最简维度并不影响计算，如果A不是最简维度，那么坐标可以从最简维度转化而来



## Complement

按一个layout (complement) 重复当前的layout直至填满shape，在这里shape一般是一维，即便是多维也会在函数处理中被视作一维，一般是`size(a)`。在代码中静态的layout和shape更易计算

### 直观地找complement layout

- shape
  1. 先重复当前的layout，使其重复成一个整块，即一维
  2. 重复刚刚的整块，即二维
- stride
  1. 刚刚重复成整块时每次重复跨过的index
  2. 重复整块跨过的index（整块index宽）



## Division

必须根据composition的左分配律，理解下面这行的几何意义

```cpp
composition(layout, make_layout(tiler, complement(tiler, size(layout))));
```

tiler是shape比原layout小，且可以重复填满原layout的layout



## Product

tiler是原layout的重复模式，tiler的stride乘以原layout的shape就是tiler间的stride，tiler的shape就是各mode方向上的重复次数



# Tensor

## partitioning

### inner and outer partitioning

inner对应local_tile，先divide划分再slice取出，一般用于partition出一个blockIdx对应的tile

outer对应local_partition，把元素分配给对应坐标

详细用法见GEMM tutuorial

## 相关API

### `local_tile`

这一步从DRAM的全局Tensor划分出属于CTA的数据块，可能是把离散的数据划分到了一块里，`gA`

两种call方式

- ```cpp
   Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
   ```

- ```cpp
   Tensor gA = local_tile(mA, select<0,2>(cta_tiler), select<0,2>(cta_coord));
   ```

是两步操作的综合

1. ```cpp
   // 先把指代DRAM中数据的Tensor根据tiler划分
   // ((BLK_M,BLK_K),(m,k))
   Tensor gA_mk = zipped_divide(mA, select<0,2>(cta_tiler));
   ```

2. ```cpp
   // 再根据coord选中这cta对应的数据块
   // (BLK_M,BLK_K,k)
   Tensor gA = gA_mk(make_coord(_,_), select<0,2>(cta_coord));
   ```

### `local_partition`

与`local_tile`相同的是，先对输入Tensor和Tiler做`zipped_divide`，得到((Tile_M, Tile_N), (Grid_M, Grid_N))形状的Tensor

根据文档的示意图，`local_tile`根据coord取一列，即整个tile

而`local_partition`取一行，即每个tile中对应的一个元素


# mma_atom

```cpp
// cutlass/include/cute/atom/mma_atom.hpp

template <class... Args>
struct MMA_Atom;
// 不完整的类声明，为specialization做准备

template <class MMAOperation>
struct MMA_Atom<MMAOperation> : MMA_Atom<MMA_Traits<MMAOperation>>
{};
// 定义在specialization前，作用是声明一条“捷径”或者是redirect
// 用户使用时可以直接把MMAOperation作为模板参数，省去MMA_Traits
// 但MMA_Traits在设计中是需要的，用于分离软件接口和硬件属性的设计，便于协作

template <class MMAOperation, class... Args>
struct MMA_Atom<MMA_Traits<MMAOperation, Args...>>
  : MMA_Traits<MMAOperation, Args...>
// 具体的specialization
```



# gemm_tutorial

## sgemm1.cu

该例主要展示了`local_tile()`和`local_partition()`的组合使用效果

核心kernel`gemm_device()`中，不论是NT或TN矩阵乘，ABC的shape始终是`(M, K) (K, N) (M, N)`，具体的col-major或row-major由stride决定。

从gmem到smem的copy调用了`local_tile()`，`local_partition()`和`copy()`。首先`local_tile()`把shape为`(M, K)`的矩阵A划分为`(BLK_M, BLK_K)`的小块，再从中选出了coord为`(blockIdx.x, _)`的所有小块，视觉上体现为一条`(BLK_M, (BLK_K, k))`“横带”，这条横带上有k个小块，对应main loop里的k次循环，在CuTe中k被当做独立的维度。矩阵B则是被选出了一条“竖带”，但将其转置，得到`(BLK_N, BLK_K, k)`的横带。矩阵C只需按照tiled gemm划分出对应的`(BLK_M, BLK_N)`。

这里local_tile输入二维的Tensor和tiler输出了三维Tensor，可以看到后两维是`((BLK_K, k))` flatten的结果，实际上这和输入的coord有关，coord中含`_`被视为slicing操作，CuTe的设计使tiled gemm更易写

经过了这一步的tiling，NT和TN矩阵乘在smem具有了一样的形式，也使得partition过程的形状统一。虽然形状上有统一形式，但在输入是NT和TN时，这样可以保证取到smem和register的gmem数据能得到正确结果吗？可以观察到NT和TN矩阵乘的区别是sA sB tA tB的stride从默认的`LayoutLeft{}`变成了`LayoutRight{}`，这样在mA mB转置的情况下，每一步划分都转置，那么得到的就是原位置的数据。但tC并没有转置，是否，恰巧是因为是正方形？在两次转置后，寄存器里的数据是原样的，结果是原样的。







# C++ meta programming

### 函数declaration中，虽然静态最优，但仍传入instance

比如`sgemm1.cu`中

```cpp
template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class AThreadLayout,
          class TB, class BStride, class BSmemLayout, class BThreadLayout,
          class TC, class CStride, class CSmemLayout, class CThreadLayout,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(CThreadLayout{}))::value)
void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
            TA const* A, AStride dA, ASmemLayout sA_layout, AThreadLayout tA,
            TB const* B, BStride dB, BSmemLayout sB_layout, BThreadLayout tB,
            TC      * C, CStride dC, CSmemLayout          , CThreadLayout tC,
            Alpha alpha, Beta beta)
```

或者是layout的创建

```cpp
make_layout(make_shape(Int<32>{}, Int<8>{}))
```

利用了C++的type deduction，实现了两个优点：

- 使得函数能同时处理动态静态情况，维护方便
- 省去了调用函数时冗长的显式模板参数

在`CUTE_STATIC_ASSERT_V`中常常能看到类似的value-based metaprogramming，如

```cpp
CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler));
```

表面上是运行期比较两个value，但实际上在编译期有另外一套设计，比较的是整数type（形如`Int<128>`）
