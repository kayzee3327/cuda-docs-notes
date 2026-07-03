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

> Given a layout, partitioning is a functional composition followed by slicing.  	--- GPU Mode Lec57 CuTe

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

## sgemm_1.cu

该例主要展示了`local_tile()`和`local_partition()`的组合使用效果

核心kernel`gemm_device()`中，不论是NT或TN矩阵乘，ABC的shape始终是`(M, K) (K, N) (M, N)`，具体的col-major或row-major由stride决定。

从gmem到smem的copy调用了`local_tile()`，`local_partition()`和`copy()`。首先`local_tile()`把shape为`(M, K)`的矩阵A划分为`(BLK_M, BLK_K)`的小块，再从中选出了coord为`(blockIdx.x, _)`的所有小块，视觉上体现为一条`(BLK_M, (BLK_K, k))`“横带”，这条横带上有k个小块，对应main loop里的k次循环，在CuTe中k被当做独立的维度。矩阵B则是被选出了一条“竖带”，但将其转置，得到`(BLK_N, BLK_K, k)`的横带。矩阵C只需按照tiled gemm划分出对应的`(BLK_M, BLK_N)`。

这里local_tile输入二维的Tensor和tiler输出了三维Tensor，可以看到后两维是`((BLK_K, k))` flatten的结果，实际上这和输入的coord有关，coord中含`_`被视为slicing操作，CuTe的设计使tiled gemm更易写

经过了这一步的tiling，NT和TN矩阵乘在smem具有了一样的形式，也使得partition过程的形状统一。虽然形状上有统一形式，但在输入是NT和TN时，这样可以保证取到smem和register的gmem数据能得到正确结果吗？可以观察到NT和TN矩阵乘的区别是sA sB tA tB的stride从默认的`LayoutLeft{}`变成了`LayoutRight{}`，这样在mA mB转置的情况下，每一步划分都转置，那么得到的就是原位置的数据。但tC并没有转置，是否，恰巧是因为是正方形？在两次转置后，寄存器里的数据是原样的，结果是原样的。

### BLAS 的 NT/TN/NN/TN 和 CuTe 的 M-major/N-major/K-major

cute文档中有这样一张表

| BLAS | A Majorness | A Layout        | B Majorness | B Layout        |
| ---- | ----------- | --------------- | ----------- | --------------- |
| NT   | M-major     | `(M,K):(1,ldA)` | N-major     | `(N,K):(1,ldB)` |
| TN   | K-major     | `(M,K):(ldA,1)` | K-major     | `(N,K):(ldB,1)` |
| NN   | M-major     | `(M,K):(1,ldA)` | K-major     | `(N,K):(ldB,1)` |
| TT   | K-major     | `(M,K):(ldA,1)` | N-major     | `(N,K):(1,ldB)` |

阅读时需要一些思维转换：

- 我们思考的出发点是矩阵在内存中的排布，不考虑加快访存的特殊排布，通常只可能

  - 矩阵每行contiguous，A的K维连续或B的N维连续
  - 矩阵每列contiguous，A的M维连续或B的K维连续

  尝试把这两种和BLAS、CuTe的表示对应起来

#### BLAS的视角

首先要理解BLAS的访存方式和默认假设。BLAS的表示中，数据的储存从公式$C = \alpha \text{op}(A) \text{op}(B) + \beta C$出发，假设

- 进行的运算始终是`M*K@K*N`
- **数据按column-major储存**。

当没有转置时，A和B的列连续，在内存中存储为每M个元素连续和每K个元素连续。

BLAS在内存中访问矩阵的某一元素$A[i, j]$时，内存按1D排布，通过以下公式找到$[i,j]$的物理index
$$
\text{Memory Index} = i + (j \times \text{lda})
$$
其中$i$在一列中平移到正确位置，范围是$[0,lda)$，$j\times lda$在各列中跳转到正确列。

当我们想要使用row-major数据时传入`CUBLAS_OP_T`类似标签，告诉BLAS我们想要物理内存里的数据的转置，lda/ldb/ldc不随是否转置变化，lda/ldb/ldc实际上只是一个stride，连续的一行始终是一个整体，跳转到下一行/列需要的元素数相同，即连续整体的元素数，除非希望矩阵形状改变，不然lda/ldb/ldc不变。

但又需要注意的是这里lda/ldb/ldc不变是相对物理内存里数据不变而言的，当处理不同矩阵时，如果需要运行时判断需不需要取该矩阵的转置，那么lda/ldb/ldc就是变化的。

有相关的小技巧可以让我们快速处理row-major数据：

- 如果不标出`CUBLAS_OP_T`这样的标签，BLAS访问数据仍按照column-major的映射公式，相当于计算中使用的是$A^\top$而不是$A$。此时我们可以利用公式$C^\top = (A \times B)^\top = B^\top \times A^\top$，交换A和B的输入位置，让BLAS读到的是$B^\top$和$A^T$,又因为$C^\top$会按column-major的逻辑写入，那么得到的就是row-major的$C$！一些传入参数的更改说明了我们变换的是逻辑和物理的映射：

  - `lda/ldb/ldc`在此处不再是跳转到逻辑上下一列的距离，而是下一行
  - `m=M, n=N, k=K`更改为`m=N, n=M, k=K`：`m`是$C^\top$的行数，`n`是$C^\top$的列数，`k`是始终不变的inner dimension。

  经过这两个更改，访问row-major数据的extent和stride都对上了

那么在动态判断需不需要取该矩阵的转置时，我们可以把连续的维度当成矩阵行来思考。

#### CuTe视角

CuTe希望能减轻用户处理不同排布数据的编程负担，比如不再有row-major和column-major储存数据的不同实现。从CuTe程序的设计理念来看，host代码处理不同物理排布，device代码使用统一流程，复杂的映射过程被编译器隐藏，而不作为显式的device代码需要用户手写。

虽然CuTe表示直白了许多，只需要shape描述逻辑上期待的形状，layout桥接了逻辑形状和物理内存的排布，就可以表达几乎所有物理排布，但出于以上设计理念，CuTe仍希望用户能满足形状假设`M*K`和`N*K`，或者`K*M`和`K*N`，且使用row-major数据，这样更有助于向量化访存、使用统一流程等。

#### 比较

上表中不难理解：

- 矩阵A的 M-major/K-major 与BLAS中 N/T 相同，内存中是 M/K 维元素连续

- 矩阵B的 N-major/K-major 与BLAS中 T/N 相同，内存中是 N/K 维元素连续

当在CuTe的语境里看到N/T时，先想象其对应BLAS的内存排布（在哪一维连续），再用stride把内存对应到`M*K`和`N*K`的形状里。



## sgemm_2.cu

该例旨在介绍更细颗粒度的`TiledCopy`和`TiledMMA`，相比于`local_partition()`，数据的layout可以适应更多形式，比如放进Tensor Core

`TiledCopy`在根据`Copy_Atom`，thread layout，value layout划分数据块之后（编译期计算），每个线程如下找到对应搬运数据：

- `ThrCopy{} = TiledCopy{}.get_slice(threadIdx.x)`
- `ThrCopy{}.partition_S()`得到src对应的Tensor
- `ThrCopy{}.partition_D()`得到dst对应的Tensor

拷贝过程中，先搬运到rmem的fragment再搬运到smem。

以`tAgA`，`tAsA`为例，取M=N=5120，K=4096，bM=bN=128，bK=8时，打印两者layout：

```shell
((_4,_1),_1,_1,512):((4096,_0),_0,_0,_8) #tAgA
((_4,_1),_1,_1):((_1,_0),_0,_0)			#tAsA
```

此时的`TiledCopy`定义是

```cpp
TiledCopy copyA = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, float>{},
                                  Layout<Shape<_32,_8>>{},  // thread
                                  Layout<Shape<_4, _1>>{}); // value
```

线程块设置为了`TiledCopy`和`TiledMMA`的尺寸256，这里线程安排为`_32,_8`，每个线程安排了`_4, _1`个元素，来划分`_128,_8`的tile。可以观察到`(CPY, CPY_M, CPY_K)`是`((_4,_1),_1,_1)`，其中`(_4,_1)`是value shape，`(CPY_M, CPY_K)`则是value shape的重复方式，即每个线程可能处理离散且重复的pattern。`tAgA`有额外一维`512`是从`gA`继承，即`K/bK`

`TiledMMA`需要一个三维layout，即M，N，K。可把`MMA_Atom<MMAOp>`换成`MMAOp`

```cpp
TiledMMA mmaC = make_tiled_mma(MMA_Atom<UniversalFMA<float>>{},
                               Layout<Shape<_16, _16, _1>>{});
```

`TiledMMA`与`TiledCopy`逻辑类似，以A为例，在划分工作到线程后得到shape`(MMA, MMA_M, MMA_K)`，此例sgemm的最小操作只需一个元素，`(MMA)`是`(_1)`，当使用Tensor Core或其他复杂硬件时，`(MMA)`就会嵌套的成为多维

## sgemm_sm80.cu (sgemm部分)

引入了流水线

tiled copy不同

```cpp
TiledCopy copyA = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, float>{},

                                      Layout<Shape<_32,_8>>{},

                                      Layout<Shape<_4, _1>>{});

TiledCopy copyA = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<TA>, TA>{},

                                      Layout<Shape<_32, _8>, Stride<_8,_1>>{},

                                      Layout<Shape<_1, _1>>{});
```

- `UniversalCopy`以寄存器为中转，使用标准拷贝指令`ld.global`和`st.shared`
- 在sm80上使用了`cp.async`指令，节省了到寄存器的开销，且`cp.async`非阻塞的拷贝是软件流水线的基础，线程在发起拷贝指令后可以继续执行，直到需要对应数据的地方再等待，可以通过计算来掩盖内存开销

### `cp.async`

LDGSTS的劣势：

1. **寄存器压力**：消耗大量寄存器来暂存数据，降低了 Active Warps（占用率 Occupancy）。

2. **指令开销**：需要发射 `LDG`（Load Global）和 `STS`（Store Shared）两条指令，占用指令发射带宽。

3. **功耗与 RF 带宽**：对寄存器文件进行读写会消耗极高的功耗，并占用 RF 的数据带宽。

4. **同步开销**：必须通过代码等待 LDG 完成（消耗显式同步或依赖隐式寄存器记分板），然后才能执行 STS，难以深度隐藏访存延迟。

`cp.async`主要由以下三个指令组合使用

1. 发起拷贝：`cp.async`

```cpp
cp.async{.ca|.cg}.shared.global [%dst], [%src], %cp_size, %src_size;
```

CuTe里以Copy Atom的copy operation实现

2. 提交异步组：`cp.async.commit_group`

CuTe里对应`cp_async_fence()`

3. 同步与等待：`cp.async.wait_group` 和 `cp.async.wait_all`

CuTe里对应`cp_async_wait<N>()`，当N==0时是`cp.async.wait_all`，其他是`cp.async.wait_group`

### Thr Layout和 Val Layout

之前的thread layout和value layout都没有stride，可以很直观地结合thread layout的排布和value layout的重复来想象出src和dst的mapping。那更复杂的情况呢？（Gemini暂时写不出好例子，多看几个例子）



### 使用`cute::ArrayEngine`在smem储存数据

空间开销没变，但是可以编译时灵活地判断类型，拥有更多接口



### 流水线设计

流水线做了两次掩盖

- Copy from gmem->smem can overlap with copies from smem->rmem and compute on rmem.
- Copy from smem->rmem can overlap with compute on rmem.

这个流水线需要如下流程

- rmem流水线没有耗尽时：从smem拷贝数据到rmem，覆盖之前已经运算过的rmem。注意拷贝到rmem的数据不是整个`(_1,_8,_8)`的块，而是其中一个小段比如`(_,_,_0)`，即在`MMA_K`上循环，每个线程负责几轮运算和几个寄存器
- rmem流水线即将耗尽时：开始处理新的tile，但同时确保流水线上没有太多在同时拷贝导致缺数据。这里设计是`cp_async_wait<K_PIPE_MAX-2>();`，结合我们预取了`K_PIPE_MAX-1`块tile，那么就是保证处理完一块tile的时间掩盖了搬运一块tile的时间，始终有一块搬运完的tile在等待
- 处理新的tile，rmem流水线刚开始时：预取下一块tile，后移当前处理块的标记

#### gmem -> smem

在预取时，最后一块pipe的smem没有取，目的有

- 更简洁的主循环，不用在开始的时候写特判
- 始终有一阶段在取

#### smem -> rmem & rmem computing

TiledMMA定义：

```cpp
TiledMMA mmaC = make_tiled_mma(UniversalFMA<TC,TA,TB>{},
                                 Layout<Shape<_16,_16,_1>>{});  // 16x16x1 TiledMMA
```

每个线程负责的`tCrA`和`tCrB`的layout是 `(_1,_8,_8):(_0,_1,_8)`，考虑到`sA`和`sB`的shape是`(_128,_8)`，线程的分工是（（（）））

寄存器的预取从一块tile到达smem开始，即`cp_async_wait<K_PIPE_MAX-2>();`，因为我们预取了`K_PIPE_MAX-1`块tile。这里需要`__syncthreads();`因为接下来的操作需要线程协同，不能有任何线程仍在搬运gmem->smem。



### TiledMMA和寄存器：`partition_A|B|C`和`make_fragment_A|B|C`

`sgemm_sm80.cu`的例子，此处正在为pipeline启动分配数据搬运

```text
gC:   gmem_ptr[32b](0x50ec00000) o (_128,_128):(_1,5120)
tCgC: gmem_ptr[32b](0x50ec00000) o (_1,_8,_8):(_0,_16,81920)
sA(_, _, 0):smem_ptr[32b](0x7c55c9000000) o ((_128,_1),(_8,_1)):((_1,_0),(_128,_0))
sB(_, _, 0):smem_ptr[32b](0x7c55c9003000) o ((_128,_1),(_8,_1)):((_1,_0),(_128,_0))
tCrA: ptr[32b](0x7c55c7fff9e0) o (_1,_8,_8):(_0,_1,_8)
tCrB: ptr[32b](0x7c55c7fffae0) o (_1,_8,_8):(_0,_1,_8)
tCrC: ptr[32b](0x7c55c7fffbe0) o (_1,_8,_8):(_0,_1,_8)
```

`partition`负责为每个thread切分好GMEM或SMEM的Tensor，返回一个仅包含thread对应element的Tensor，可以发现`gC`和`tCgC`，但是layout变化成更小shape更离散stride，即该线程在`gC`中对应的数据。

`make_fragment`负责为这些数据分配寄存器空间，可以看到`tCrC`的指针位置与`tCgC`大不相同，而与`tCrA`相似，证明是寄存器地址。

#### 不止分配内存

在CuTe中，`partition_A|B|C`和`make_fragment_A|B|C`需要绑定使用，`make_fragment`的输入需要是`partition`的输出，因此诞生了`partition_fragment_A`这样的接口：当你不需要`partition`的输出如`tCgC`时，两步操作可以一步完成。

如果`make_fragment`只负责分配寄存器空间，那为什么需要和`partition`的输出绑定呢？

正如注释所说 

> The reasoning is that we can inspect the layout of the partitioned data and attempt to match it in generated fragment to promote vectorization when copying from partition to fragment.

这两个操作绑定使得可以根据`partition`的输出来更好地设计`make_fragment`，尽可能实现向量化拷贝。



### Copy Atom Retiling

在上一个例子`sgemm_2.cu`中，显式控制了从GMEM到SMEM的拷贝，即g-r-s的LDGSTS。在本例中涉及到流水线和异步拷贝，控制更加细化：

- 上一例中，传入`gemm()`的是SMEM的`Tensor`，`tCsA`和`tCsB`，从SMEM到寄存器这一步被省略。本例中，新加入了一个`Copy_Atom`，负责将SMEM的数据映射到寄存器
- 虽然在这个sgemm中，在SMEM和寄存器的数据排布不需要变换，但实际上指出了拷贝到SMEM和最终运算的数据排布仍可能有一层变换，比如使用Tensor Core等硬件时需要多维数据而非一维。

所以可以看到例子中：

```cpp
Tensor tXsA = s2r_thr_copy_a.partition_S(sA); 	// (CPY,MMA_M,MMA_K,PIPE)
Tensor tXrA = s2r_thr_copy_a.retile_D(tCrA);	// (CPY,MMA_M,MMA_K)
```

对于`sA`的线性排列内存只需partition，而`tCrA`需要retile

这里`tCrA`和`tXrA`的职责需要区分，看一组例子：

```
tCrA : ptr[32b](0x72d057fffae0) o (_1,_8,_8):(_0,_1,_8)
tXsA : smem_ptr[32b](0x72d059000000) o ((_1,_1),_8,_8,(_1,_3)):((_0,_0),_16,_128,(_0,_1024))
tXrA : ptr[32b](0x72d057fffae0) o ((_1,_1),_8,_8):((_0,_0),_1,_8)
```

`tCrA`把寄存器和逻辑排布联系在一起，而`tXrA`和`tXsA`把需要拷贝的数据划分成可以重复的tile，方便`copy()`按线程工作，由第0维可见`(_1,_1)`

所以具体分工是：

- TiledMMA以一个TV layout决定哪些数据分配给某个线程，划分好总的工作
- s2r TiledCopy告诉每个线程怎么把数据拷贝到寄存器，决定每一小份工作怎么做，具体由copy atom决定。简单向量化和LDSM retile后的视角完全不一样。



# CuTe TMA Tensors

## TMA descriptor

TMA descriptor将GMEM中的tile打包成一个多维tensor，由以下组成

- tensor的起始指针
- tensor的数据类型
- 每一维的size
- 每一维的stride
- 其他信息如smem的大小布局、swizzle方式、越界处理方式（如predication）

TMA descriptor必须执行前在host创建，它由所有执行TMA指令的线程块共享。执行TMA指令时，需要三种参数

- TMA descriptor的指针
- smem的指针
- 可以输入TMA descriptor中GMEM多维tensor的坐标

也就是说TMA只在compile-time确定的GMEM指针上操作，但GMEM需要在运行时才有可知的布局。

## Building a TMA Tensor

### Implicit CuTe Tensors

CuTe tensor由一个iterator和一个layout组成，虽然教程中常常以pointer作为iterator，让tensor看起来似乎是地址的映射，但实际上iterator可以是任何random-access iterator。

比如*counting iterator*，它是从某个整数开始逐1增长的iterator，不存在于真正的内存中，而只是根据layout和coordinate找到对应的整数，在编译期它会被直接运算成整数。即使是如此简单的iterator，所有layout操作都可以用在它身上，记住layout是从坐标到offset的函数。

```cpp
Tensor A = make_tensor(counting_iterator<int>(42), make_shape(4,5));
```

而TMA指令不需要映射到地址或是整数，它需要映射到一个GMEM的坐标，让TMA能在运行时根据传入的tensor找到搬运的数据。

### ArithTupleIterators and ArithTuples

可以把`ArithTupleIterators`看作是多维的`counting_iterator`，这种iterator对应的值是一个坐标，offset是`ArithTuple`，所谓`ArithTuple`也只是为`IntTuple`重载了+运算，使得offset可以加到base上。在CuTe中，`IntTuple`的相加利用了`Int<>`相加的特性，类和整数可以混合着相加。

### Strides aren’t just integers

常用的tensor把坐标映射到地址或者说一个整数，实际上是坐标和stride的内积。拓展到TMA的情景下，这种内积需要变成多维的矩阵内积，最终得到一个多维的结果，即`ArithmeticTuple`，此时需要重载\*运算。

#### Aside: Integer-module strides

> A group of objects that support addition between elements and product between elements and integers is called an integer-module.

integer-module是具有 `Z*M -> M`的阿贝尔群

#### Basis elements

#### Linear combinations of strides

# C++ meta programming（补充）

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
