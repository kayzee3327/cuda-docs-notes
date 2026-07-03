# 异步拷贝机制，cp.async，mbarrier

https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-asynchronous-copy

异步拷贝的bulk拷贝需要是16B的倍数

## [Completion Mechanisms for Asynchronous Copy Operations](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-asynchronous-copy-completion-mechanisms)

一个线程需要显式地等到异步拷贝的completion。异步拷贝发起后结束前，任何对对应显存的修改都是UB。

异步拷贝在ptx中自Ampere架构引入了两种机制支持，Async-group机制和mbarrier-based机制，且在后续架构中得到了发展，每个机制有对应的指令集和架构且不能任意选择，例如hopper上使用了TMA的`cp.async.bulk`必须与mbarrier配合。

### [Async-group mechanism](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-asynchronous-copy-completion-mechanisms-async-group)

这个机制分为四步，每个线程独立进行。

1. 发起issue：一个线程发起许多异步拷贝指令
2. 提交commit：线程发起commit操作创建一个async-group，由线程私有，包含了先前所有未commit的异步拷贝，即每个异步拷贝只会存在于一个async-group里
   - 因为底部调用硬件的不同，bulk异步拷贝和非bulk异步拷贝需要放在不同的async-group里
   - 每个线程的async-group会严格按照commit顺序结束，但每个async-group里的异步拷贝是无序的
3. 等待wait：等待一个async-group或所有该线程发起的异步拷贝结束
4. Once the *async-group* completes, access the results of all asynchronous operations in that *async-group*.

### [Mbarrier-based mechanism](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-asynchronous-copy-completion-mechanisms-mbarrier)（未完成）

`mbarrier`是一个对象，可以用mbarrier的不同phase来追踪多个异步操作。mbarrier必须在bulk操作中标明，非bulk操作可以在执行后标明

1. Initiate the asynchronous operations.
2. Set up an *mbarrier object* to track the asynchronous operations in its current phase, either as part of the asynchronous operation or as a separate operation.
3. Wait for the *mbarrier object* to complete its current phase using `mbarrier.test_wait` or `mbarrier.try_wait`.
4. Once the `mbarrier.test_wait` or `mbarrier.try_wait` operation returns `True`, access the results of the asynchronous operations tracked by the *mbarrier object*.

##  [Async Proxy](https://docs.nvidia.com/cuda/parallel-thread-execution/#async-proxy) （未完成）



## [Data Movement and Conversion Instructions: Non-bulk copy](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-non-bulk-copy)

### `cp.async`

- 非阻塞异步拷贝，把数据从global`src`搬运到shared`dst`
- 数据大小`cp-size`只能是4，8，16B
- 可以用一个int32`src-size`指定拷贝最大数据量，超过该量的在`dst`用0填满，且该量不能超过`cp-size`，否则UB
- 可以标明一个`ignore-src`来将`dst`填满0，可用在predicate中

```cpp
cp.async.ca.shared{::cta}.global{.level::cache_hint}{.level::prefetch_size}
                         [dst], [src], cp-size{, src-size}{, cache-policy} ;
cp.async.cg.shared{::cta}.global{.level::cache_hint}{.level::prefetch_size}
                         [dst], [src], 16{, src-size}{, cache-policy} ;
cp.async.ca.shared{::cta}.global{.level::cache_hint}{.level::prefetch_size}
                         [dst], [src], cp-size{, ignore-src}{, cache-policy} ;
cp.async.cg.shared{::cta}.global{.level::cache_hint}{.level::prefetch_size}
                         [dst], [src], 16{, ignore-src}{, cache-policy} ;

.level::cache_hint =     { .L2::cache_hint }
.level::prefetch_size =  { .L2::64B, .L2::128B, .L2::256B }
cp-size =                { 4, 8, 16 }
```

#### `.cg`和`.ca`qualifier（[Cache Operators](https://docs.nvidia.com/cuda/parallel-thread-execution/#cache-operators)）

Memory load

| qualifier | 行为                                                         |
| --------- | ------------------------------------------------------------ |
| `.ca`     | 在L1 L2中都缓存，针对很快就要再次访问的数据、smem里不储存的数据 |
| `.cg`     | 不在L1中缓存                                                 |
| `.cs`     |                                                              |
| `.lu`     |                                                              |
| `.cv`     |                                                              |

memory store

| qualifier | 行为 |
| --------- | ---- |
| `.wb`     |      |
| `.cg`     |      |
| `.cs`     |      |
| `.wt`     |      |

观察`cp.async`的syntax

- `cp.async.cg`的`cp-size`只能使用16，CuTe中也有`static_assert(sizeof(TS) == 16, "cp.async sizeof(TS) is not supported")`侧面印证。
- `cp.async.ca`的`cp-size`更灵活

##### 缓存方式：GLOBAL还是ALWAYS

GPU的内存控制器（Memory Controller）是以 **32 字节（32 Bytes Sector）** 为基础粒度进行内存事务处理的，L1/L2 cache的最小granularity为32byte或一个sector，一个cache line由128byte即4 sector组成。L2cache的精细度是可调整的。

如果 32 个线程每个只读取 4 字节，只要这 128 byte在物理内存上是连续的（Coalesced），硬件只需发起 4 个 32 字节的内存事务即可，如果不连续，那么LSU也能精准地找到对应的内存块。既然 GMEM 本身支持 4 字节读取，为什么 `cp.async.cg` 不行？以下从**没有TMA**的视角解释。

首先，cache always时，异步拷贝的过程大致如下

1. SM向其LSU发起异步拷贝指令`cp.async.ca`
2. memory controller从VRAM中取出32字节为单位的数据传递给L2 cache（如果L1/L2中没有）
3. 数据从L2缓存到L1，如果L1缓存有效，L1会优先命中
4. LSU指挥L1中的数据（32 byte或sector为单位的cache line），按需要搬运到smem（4，8，16byte）

整个过程不需要register参与。再看cache global的流程

1. 同样SM向其LSU发起异步拷贝指令`cp.async.cg`，请求直接转向L2
2. memory controller从VRAM中取出32字节为单位的数据传递给L2 cache（如果L2没有，不查找L1）
3. L2 cache直接把数据转交给smem子系统，按bank写入

cache global实际上提供了一条快速拷贝的通道。cache always有一套复杂的系统（LSU，L1）来处理各种零散的内存访问以及优化（如coalesce访问），选择cache global意味着跳过这些步骤，但需要开发者精确管理，以满足从L2到smem的大块内存搬运。

另外选择固定16byte，一个线程单条指令所能请求的最大显存，即一个warp 512byte，符合128byte的cache line大小，便于利用最大带宽。

##### Ampere的不足

但即便是利用了最大带宽，一次拷贝的显存始终受限，遇到大块内存访问，仍需要消耗很多SM资源来发射拷贝指令。在hopper中引入TMA解决了这一问题，`cp.async.bulk`相关指令可以达到数十KB

#### `.shared` qualifer（[Shared State Space](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=shared#shared-state-space)）

> The shared (`.shared`) state space is a memory that is owned by an executing CTA and is accessible to the threads of all the CTAs within a cluster

一般指SMEM或DSMEM，有两种sub-qualifier，`::cta`和`::cluster`，默认是`::cta`，他们表示运算的地址是属于当前线程块或cluster的。在hopper以前的架构中，`.shared`就是指当前cta可访问的SMEM，向下兼容。

- 例如`.global`/`.local`/`shared::cta`/`shared::cluster`都有对应的address window，即在虚拟地址空间里的特定地址段。`shared::cta`地址可以通过线程块ID映射到`shared::cluster`地址，后者包含前者
- 区分DSMEM的地址段可以在cluster通信时减少开销
- 也使标注`::cta`的地址操作不会增加开销

#### `.global` qualifier（[Global State Space](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=shared#global-state-space)）

> The global (`.global`) state space is memory that is accessible by all threads in a context.

软件上描述的是所有线程块（CTA）、cluster、grid等都能通信的内存位置，硬件上体现为GMEM

#### `.level::prefetch_size` qualifier

这是一个hint，硬件在缓存压力较大或内存带宽用满时可能选择性忽略，将prefetch_size大小（64，128，256B）的数据额外取到对应缓存。这是通常和`.global` qualifier一起用的。

显存（HBM或GDDR）被访问时会打开一个buffer，被称为一个page或row buffer，对应GMEM的一个128byte的连续段，打开这个buffer开销很大，再次访问buffer里的数据会很快（这也是coalesce访问的硬件基础）。以GEMM为例，在K维上循环取列时，每列之间的数据没法coalesce，此时如果预取到L2，就能减少很多打开buffer的开销。

CuTe中常常选择128B作为预取大小，不仅仅因为这符合cache line大小，而且更大或更小更可能造成缓存的浪费（Cache Thrashing），另外矩阵乘中coalesce访问使得预取过大没有意义。



### `cp.async.commit_group`

> `cp.async.commit_group` instruction creates a new *cp.async-group* per thread and batches all prior `cp.async` instructions initiated by the executing thread but not committed to any *cp.async-group* into the new *cp.async-group*. If there are no uncommitted `cp.async` instructions then `cp.async.commit_group` results in an empty *cp.async-group.*



### `cp.async.wait_group`**,** `cp.async.wait_all`

> `cp.async.wait_group` instruction will cause executing thread to wait till only `N` or fewer of the most recent *cp.async-group*s are pending and all the prior *cp.async-group*s committed by the executing threads are complete.

等待，直到还有N个async-group在拷贝，因为async-group之间是严格按照commit顺序的，所以这个指令可以等到任意async-group结束。



### `cp.async` 的写操作可见时机

- 对应`cp.async.wait_group`**,** `cp.async.wait_all`完成
- 或`mbarrier.test_wait`通过



## [Data Movement and Conversion Instructions: Bulk copy](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-bulk-copy)

