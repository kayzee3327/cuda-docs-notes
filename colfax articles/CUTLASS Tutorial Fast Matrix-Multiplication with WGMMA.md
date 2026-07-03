# Asynchronous Warpgroup MMA (WGMMA)

两种运算

- `C = A * B + C`
- `C = A * B`, where the input from accumulator `C` is disabled.

数据位置要求

- A: either SMEM or register memory (RMEM)
- B: always be stored in shared memory (SMEM)
- C: always held in RMEM

# WGMMA inside a CUTLASS kernel

- `cute::warpgroup_arrive();` - `wgmma.fence.sync.aligned`

- `cute::gemm()`

- `cute::warpgroup_commit_batch();` - `wgmma.commit_group.sync.aligned`

- `cute::warpgroup_wait<0>();` - `wgmma.wait_group.sync.aligned`

## MMA_Atom

以NT hgemm为例，

```cpp
TiledMMA tiled_mma = cute::make_tiled_mma(
  SM90_64x64x16_F16F16F16_SS<GMMA::Major::MN,GMMA::Major::MN>{});
```

对应`wgmma.mma_async.sync.aligned.m64n64k16.f16.f16.f16`

- MxNxK 64x64x16表示wgmma的计算size
- SS或RS表示AB的数据位置SMEM/RMEM
- 模板参数表示MN-major或K-major，非16b operand数据类型必须使用K-major

# SMEM layout constraints for WGMMA





# Synchronization for WGMMA

