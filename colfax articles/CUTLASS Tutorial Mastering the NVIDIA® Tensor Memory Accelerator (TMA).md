# 提要

## TMA带来的的优势

- 将线程从异步拷贝中解放出来，以warp-specialization调度拷贝计算的异步操作，提高GPU使用率
- 通过TMA copy descriptor转移地址计算和predication计算（比如边界检查）的overhead给TMA

## 核心指令和对象

- [`cp.async.bulk.tensor`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor)
- [`cp.reduce.async.bulk.tensor`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-reduce-async-bulk-tensor)
- [cuTensorMap](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html) operand
- async memory barrier (i.e., [`mbarrier`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier))
- async memory fence (i.e., [`fence.proxy.async`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar-fence))

# TMA Load

使用TMA Load的kernel与使用内存拷贝（LSU）不同。

## Host Code

```cpp
template <typename T, int CTA_M, int CTA_N>
void host_fn(T* data, int M, int N) {
  using namespace cute;
 
  // create the GMEM tensor
  auto gmem_layout = make_layout(make_shape(M, N), LayoutRight{});
  auto gmem_tensor = make_tensor(make_gmem_ptr(T), gmem_layout);
 
  // create the SMEM layout
  auto smem_layout = make_layout(make_shape(CTA_M, CTA_N), LayoutRight{});
 
  // create the TMA object
  auto tma_load = make_tma_copy(SM90_TMA_LOAD{}, gmem_tensor, smem_layout);
 
  // invoke the kernel
  tma_load_kernel<CTA_M, CTA_N>
                 <<<dim3{M / CTA_M, N / CTA_N, 1}, 1>>>
                 (tma_load, gmem_tensor, smem_layout);
}
```

`tma_load`本身依然是Copy_Atom组成的TiledCopy，但其Copy_Atom在底层是由`gmem_tensor` and `smem_layout`（含有smem layout和swizzle信息）构造的`TmaDescriptor`，an alias for [CUtensorMap](https://github.com/NVIDIA/cutlass/blob/637b15906358191cb4238af419d408a65819d7ec/include/cute/arch/copy_sm90_desc.hpp#L178)。`SM90_TMA_LOAD{}`对应`cp.async.bulk.tensor` PTX call。

在实际应用中，每个GMEM tensor都需要创建一个对应的`TmaDescriptor`，这个过程必须在kernel launch之间完成，也就是TiledCopy创建时。



## Kernel Code

```cpp
template <typename T, int CTA_M, int CTA_N, class TmaLoad, class GmemTensor>
void tma_load_kernel(__grid_constant__ const TmaLoad tma_load, GmemTensor gmem_tensor) {
  using namespace cute;
  constexpr int tma_transaction_bytes = CTA_M * CTA_N * sizeof(T);
 
  __shared__ T smem_data[CTA_M * CTA_N];
  __shared__ uint64_t tma_load_mbar;
 
  auto smem_layout = make_layout(make_shape(CTA_M, CTA_N), LayoutRight{});
  auto smem_tensor = make_tensor(make_smem_ptr(smem_data), smem_layout);
 
  if (threadIdx.x == 0) {
    auto gmem_tensor_coord = tma_load.get_tma_tensor(shape(gmem_tensor));
 
    auto gmem_tensor_coord_cta = local_tile(
        gmem_tensor_coord,
        Tile<Int<CTA_M>, Int<CTA_N>>{},
        make_coord(blockIdx.x, blockIdx.y));
 
    initialize_barrier(tma_load_mbar, /* arrival count */ 1);
 
    set_barrier_transaction_bytes(tma_load_mbar, tma_transaction_bytes);
 
    auto tma_load_per_cta = tma_load.get_slice(0);
    copy(tma_load.with(tma_load_mbar),
         tma_load_per_cta.partition_S(gmem_tensor_coord_cta),
         tma_load_per_cta.partition_D(smem_tensor));
  }
  __syncthreads();
  wait_barrier(tma_load_mbar, /* phase */ 0);
 
  // after this line, the TMA load is finished
}
```

例子展示了一个cta tile的搬运，这里只需要一个线程发射指令。tma_load可以自动通过partitioning得到所需坐标，再传入内置gmem_tensor中。

函数声明中tma_load需要前缀`__grid_constant__ const`使其位于常量地址空间，与`blockIdx`相同，这实际上是对`cuTensorMap`从host到device拷贝的要求。

### Memory barrier

- `initialize_barrier` - `mbarrier.init.shared.b64` 初始化mbarrier并设定expect arrival count
- `set_barrier_transaction_bytes` - `mbarrier.arrive.expect_tx.shared::cta.b64` 在sm90中，mbarrier进化为 **Transaction-based**，新引入了tx_count来记录搬运数据的大小。TMA写入SMEM时自动增加这个计数，只有搬运了足够的数据时，mbarrier才会改变状态。
- `copy` - `cp.async.bulk.tensor` 会自动分配合适的`mbarrier::complete_tx::bytes`
- `wait_barrier` - `mbarrier.try_wait.parity.shared::cta.b64`需要前置`__syncthreads();`。只有当mbarrier重新回到状态0时，才会继续执行。try_wait意味着这是阻塞的等待

> the write to SMEM done by the TMA load is made visible to all threads that invoked the mbarrier wait



### REMAINDER TILES WITH TMA and STRIDE REQUIREMENTS

TMA自动完成地址计算且不需要考虑边界情况，硬件自动补0，这与“implicit” CuTe tensors with `ArithTuple`的目的一致

TMA的拷贝必须是整块而不是strided的，其中tile必须

- 有contiguous维 stride=1
- strided维必须以16字节为倍数

# TMA Store

```cpp
template <typename T, int CTA_M=32, int CTA_N=32>
void host_fn(T* data, int M, int N) {
  using namespace cute;
 
  // create the GMEM tensor
  auto gmem_layout = make_layout(make_shape(M, N), LayoutRight{});
  auto gmem_tensor = make_tensor(make_gmem_ptr(T), gmem_layout);
 
  // create the SMEM layout
  auto smem_layout = make_layout(make_shape(CTA_M, CTA_N), LayoutRight{});
 
  // create the TMA object
  auto tma_store = make_tma_copy(SM90_TMA_STORE{}, gmem_tensor, smem_layout);
 
  // invoke the kernel
  tma_store_kernel<CTA_M, CTA_N>
                  <<<dim3{M / CTA_M, N / CTA_N, 1}, CTA_M>>>
                  (tma_store, gmem_tensor, smem_layout);
}
 
template <typename T, int CTA_M, int CTA_N, class TmaStore, class GmemTensor>
void tma_store_kernel(__grid_constant__ const TmaStore tma_store, GmemTensor gmem_tensor) {
  using namespace cute;
  __shared__ T smem_data[CTA_M * CTA_N];
 
  auto smem_layout = make_layout(make_shape(CTA_M, CTA_N), LayoutRight{});
  auto smem_tensor = make_tensor(make_smem_ptr(T), smem_layout);
 
  // fill the rows of smem_data
  for (int j = 0; j < CTA_N; ++j) {
    smem_data(threadIdx.x, j) = threadIdx.x;
  }
  
  __syncthreads();
  tma_store_fence();
 
  if (threadIdx.x == 0) {
    auto gmem_tensor_coord = tma_store.get_tma_tensor(shape(gmem_tensor));
 
    auto gmem_tensor_coord_cta = local_tile(
      gmem_tensor_coord,
      Tile<Int<CTA_M>, Int<CTA_N>>{},
      make_coord(blockIdx.x, blockIdx.y));
 
    auto tma_store_per_cta = tma_store.get_slice(0);
    copy(tma_store,
         tma_store_per_cta.partition_S(smem_tensor),
         tma_store_per_cta.partition_D(gmem_tensor_coord_per_cta));
    // tma_store_arrive();
  }
  // tma_store_wait<0>();
}
```

## 不同的同步机制：*memory fence*

- `tma_store_fence()` - `fence.proxy.async.shared::cta`让整个线程块等待fence前memory access结束，因为后续操作是部分线程且异步的，需要async proxy

- `tma_store_arrive()`与`tma_store_wait<Count>()`组合也能起到同步作用，如果写出SMEM需要再写入，这个组合很适合



# A Deeper Look at TMA Operations

## TMA Store Reduce

使用`cp.reduce.async.bulk`在TMA实现SMEM写回GMEM时的reduce操作，如add max min等，节省了拷贝到寄存器再计算的开销。CuTe中实现：

```cpp
// to create a TMA reduce sum object
auto tma_reduce_sum = make_tma_copy(SM90_TMA_REDUCE_ADD{}, gmem_tensor, smem_layout);
```

## TMA Load Multicast

需要定义cluster，在同cluster内具有相同TMA Descriptor的cta请求相同数据时，multicast触发
