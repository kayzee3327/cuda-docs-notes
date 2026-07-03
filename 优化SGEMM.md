复杂系统优化技巧是无法穷尽的，怎么才能分析出提升最多的优化呢？

# 大局观

经验太少，这一部分尚在犹豫中。

我想先提出几个问题：

- 如何去系统地思考kernel优化
- 如何判断一个kernel是否成功优化

## 系统思考

[CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#overall-performance-optimization-strategies)可以先大概自作主张地分以下几步

- 暴露程序的可并行部分

- 考虑最小化内存访问

- 利用指令优化计算、内存搬运

需要了解硬件实现和pipe的相互协作，以及profiler，Quick Refs：

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html

- [Hardware Model](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#hardware-model)

- [System Architecture Guide](https://docs.nvidia.com/nsight-graphics/UserGuide/gpu-trace-system-architecture.html#sm-throughput)

- [Modal GPU Glossary](https://modal.com/gpu-glossary)

- [NV Dev Forum: Concurrency between CUDA cores and Tensor Cores](https://forums.developer.nvidia.com/t/i-need-help-understanding-how-concurrency-of-cuda-cores-and-tensor-cores-works-between-turing-and-ampere-ada/286305)

- [NV Dev Forum: Questions about MIO, LSU, L1/SMEM, and Instruction Dispatch on SM80+ Architectures](https://forums.developer.nvidia.com/t/questions-about-mio-lsu-l1-smem-and-instruction-dispatch-on-sm80-architectures/346532)

- [NV Dev Forum: How does the LSU (Load/Store Unit) execute Load/Store instructions in the Ampere architecture?](https://forums.developer.nvidia.com/t/how-does-the-lsu-load-store-unit-execute-load-store-instructions-in-the-ampere-architecture/273699)

- [NV Dev Forum: Global Load and Texture Load on LSU Traffic](https://forums.developer.nvidia.com/t/global-load-and-texture-load-on-lsu-traffic/351112)

- [NV Dev Forum: What’s the difference between MIO and LSU instruction queue in Volta architecture?](https://forums.developer.nvidia.com/t/whats-the-difference-between-mio-and-lsu-instruction-queue-in-volta-architecture/124749)

- warp uniform execution [Dissecting the NVidia Turing T4 GPU via Microbenchmarking](https://arxiv.org/pdf/1903.07486)

- [NV Dev Forum: Understanding uniform registers](https://forums.developer.nvidia.com/t/understanding-uniform-registers/347103)

- shared memory mechanism [NV Dev Forum](https://forums.developer.nvidia.com/t/requesting-clarification-for-shared-memory-bank-conflicts-and-shared-memory-access/268574/12)

- shared memory bank conflict mechanism [NV Dev Forum: LSU Wavefront Scheduling and Shared Memory Bank Utilization on Blackwell](https://forums.developer.nvidia.com/t/lsu-wavefront-scheduling-and-shared-memory-bank-utilization-on-blackwell/359791)

- wavefront [NV Dev Forum: Waht’s the difference between ‘wavefronts’ and ‘sectors/Req’?](https://forums.developer.nvidia.com/t/wahts-the-difference-between-wavefronts-and-sectors-req/165293)

- [NV Dev Forum: What is ‘uncoalesced shared accesses’](https://forums.developer.nvidia.com/t/what-is-uncoalesced-shared-accesses/303722)

- [NV Dev Forum: Question about l1tex__data_pipe_lsu_wavefronts.avg](https://forums.developer.nvidia.com/t/question-about-l1tex-data-pipe-lsu-wavefronts-avg/325024)

- LSU bound [NV Dev Forum: Same SOL for memory and SM Throughput](https://forums.developer.nvidia.com/t/same-sol-for-memory-and-sm-throughput/309092)

- experience about shared memory bottlenecks [NV Dev Forum: Shared memory bank conflicts and nsight metric](https://forums.developer.nvidia.com/t/shared-memory-bank-conflicts-and-nsight-metric/115731)

- stall wait [How to know my kernel if Pipeline parallel by nsight compute](https://forums.developer.nvidia.com/t/how-to-know-my-kernel-if-pipeline-parallel-by-nsight-compute/249106)

-  how gpu raise transactions if a warp loads more than 128 bytes [NV Dev Forum: How to understand the bank conflict of shared_mem](https://forums.developer.nvidia.com/t/how-to-understand-the-bank-conflict-of-shared-mem/260900/2)



GEMM和Profiling学习资源：

- [Hamza's blog](https://hamzaelshafie.bearblog.dev/worklog-optimising-gemm-on-nvidia-h100-for-cublas-like-performance-wip/)
- [Aleksa Gordić's blog](https://www.aleksagordic.com/blog/matmul#cpt1)
- [GPU MODE Lecture 44: NVIDIA Profiling](https://www.youtube.com/watch?v=F_BazucyCMw)
- [Lecture 37: Introduction to SASS & GPU Microarchitecture](https://www.youtube.com/watch?v=we3i5VuoPWk)
- [Lecture 45: Outperforming cuBLAS on H100](https://www.youtube.com/watch?v=ErTmTCRP1_U)
  - [文章](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog)
- [Lecture 101: Learning CUTLASS the hard way](https://www.youtube.com/watch?v=jGouxuAHIfQ)
  - 专注于后半部分，对应文章[Learn CUTLASS the hard way - part 2!](https://www.kapilsharma.dev/posts/learn-cutlass-the-hard-way-2/)
- [NVIDIA On-Demand Playlists](https://www.nvidia.com/en-us/on-demand/playlist/playList-c9450de5-2ffd-4ea9-8a1b-24aeeaf49d4e/)
- https://developer.nvidia.com/nsight-compute-videos

CuTe Refs

- [[QST] What is PermutationMNK in TiledMMA in CUTLASS 3.4 changes?](https://github.com/NVIDIA/cutlass/discussions/1345)

## 方法论

### 初步分析kernel

尝试分析kernel，尝试对优化结果kernel有所预知。例如根据时间复杂度推断出SGEMM最繁忙的pipe是FMA，或者推断出FMA和LDS指令比例等

### 宏观分析：SOL

确定bounded by what，可能是某一最繁忙的pipe，或。。。

- Compute bound (FMA/HMMA > 80%)：GPU已经尽可能快的在计算了（尝试减少不必要的计算，或者使用低精度的type等）
- Memory-Bandwidth Bound (Memory > 80%)：从DRAM到L2 Cache的物理带宽已经用满，即片上片下之间的带宽已满（增加缓存利用，compression，或低精度数据类型等）
- Instruction/Latency Bound (LSU, TEX, or Stalls > 80%)：SM在花大量时间执行非计算指令或等待

### 微观分析：提出对应pipe问题的假设

这些假设应满足硬件限制，逻辑上不遗漏问题

#### LSU Pipe繁忙（Inst-Bound/Latency-Bound）

- LD/ST总量过多：是否发射了过多的LD/ST请求，超出实际计算所需？
- LD/ST效率不够：是否使用了当前场景效率最高的指令？
- LD/ST指令replay：是否有硬件冲突（bank conflicts / ...）导致同一条指令需要更多cycle执行？

每个c++内存请求，都可以按三步看待：算法-指令-硬件执行。LSU指令过多时，

1. 看算法，搬运总量出问题：用户要求过多的搬运（低AI），或者编译器自动拓展了算法所需内存搬运（如register spilling）
2. 看指令，选择指令出问题：如向量化
3. 看硬件执行，硬件资源不能理想化执行指令：如uncoalesced / bank conflicts / ...

#### DRAM带宽用满（Mem-Bound）

- DRAM请求过多次数：低缓存利用率，uncoalesced access等
- DRAM请求冗余数据：搬运不需要的数据，或是精度过高的数据

#### Math Pipe最繁忙（Comp-Bound）

- 看指令比例是否符合预期，一般可以进行下一步分析了

<!--还有更多可以补充的-->

### Pipe调度分析（Latency-Bound）

- SM的执行指令正确，繁忙pipe正确，性能却仍不够高时；
- SM throughput和Memory throughput都比较低

研究的问题不再是SM在做什么，而是SM应该在什么时机做。常常需要检查**Warp State Statistics** section。解决方案是掩盖latency。



### 检查硬件特性

是否有硬件特性未被使用？如`cp.async`，TMA，wgmma等。先让kernel走上正确的路，再走捷径。



## 验收



# Case Study: `sgemm_2.cu` A10 PCIe GPU调优

cute官方例子`sgemm_2.cu`可能是一个不错的优化出发点，

- 它应用了基本的tiling、向量化等优化，
- 使用了基础的LDGSTS来搬运数据，
- 未使用Ampere、Hopper等特有的架构优化，如async copy，
- 暴露到最细的cute控制粒度——寄存器（`tArA`）和线程（`ThrCopy`），虽然mma阶段的控制不完整，直接交给gemm函数`gemm(mma, tCsA, tCsB, tCrC);`，但这样没有错误优化的半成品很适合实验，
- `sgemm_sm80.cu`和`sgemm_sm70.cu`都引入了流水线设计，但profile结果显示并没有更充分的利用smem和计算单元，反而速度变慢。

## `sgemm_2.cu`的优化

### 静态量展开

CuTe会在编译时把所有计算和内存搬运指令展开。

### 向量化访存

`Copy_Atom`设置为`UniversalCopy<uint128_t>`实现向量化访存

### 以寄存器为buffer的double buffering

`sgemm_2.cu`使用LDGSTS来做GMEM到SMEM的内存搬运，在`GMEM->RMEM->SMEM`的数据流中，使用了寄存器和SMEM实现了double buffering，基于Stall-On-Use架构和Instruction-Level Parallelism（ILP）实现。整体数据搬运流程有四步：

1. GMEM $\rightarrow$ 临时RMEM
2. 临时RMEM $\rightarrow$ SMEM
3. SMEM $\rightarrow$ 计算RMEM
4. 计算RMEM $\rightarrow$ 计算单元计算

当一个tile被搬运到SMEM之后，临时RMEM不必再储存同一个tile，可以开始下一个tile的搬运。所以这里形成了一个两步流水线。

#### 如何overlap

当一个tile被取到SMEM后，立刻发射搬运下一个tile的指令，由于GMEM的latency大，利用Stall-On-Use架构的特性和没有数据依赖的特点，这段时间可以和3、4两步掩盖。

#### Instruction-Level Parallelism

简单来说，ILP指的是几组没有数据依赖的指令可以并发执行。在GPU上，Thread-Level Parallelism（TLP）和ILP是两种常见的并行方式，他们之间需要区分。

GPU上有巨大数量的warp，隐藏latency（或者说执行时间）的重要方式是切换执行warp，比如当warp 1执行数据搬运时，切换warp 2执行计算，这就是TLP。但单纯依靠TLP有缺陷，

- warp总量受register file和shared memory限制，
- 虽然warp间能切换，但是遇到需要长时间等待的指令，可能不会每刻都有足够的warp来切换。

借用NCU的名词来说，GPU上存在着多种“pipe”，即有多种负责不同功能的流水线，在不同pipe上的指令可以**Dual-Issue** 或 **Multiple-Issue**，在同一pipe上的指令可以流水线执行，ILP得以实现。这样就有源源不断的指令从scheduler分发给不同的warp，使硬件尽可能繁忙。

当增加ILP时，单线程执行更多指令会使用更多寄存器，就减少了TLP的程度。所以TLP和ILP之间有trade-off。

#### Stall-On-Use架构

Stall-On-Use架构与Stall-On-Issue架构相对，前者发射完指令立刻着手下一条指令，直到数据依赖时才等待，后者则等待指令完成。

英伟达GPU基于顺序执行（In-Order Execution）设计。对于GPU单线程而言，指令始终是顺序的，通过切换warp（TLP）来掩盖。而CPU常使用的乱序执行（Out-of-Order Execution）基于大量分析指令的硬件，CPU上执行的指令逻辑可能很复杂，线程较少。为了提高效率，在微架构上则不保证没有数据依赖的指令顺序，可以自由选择已经ready的流水线阶段执行（fetch/decoding/dispatch等）。

在顺序执行的情况下，记录数据依赖（寄存器依赖）对于GPU很重要，有一个专门的Scoreboard硬件负责。

#### 总结：Stall-On-Use实现ILP，ILP减少Stall，编译器负责schedule来充分利用

## `gemm()`内部的优化

在`include/cute/algorithm/gemm.hpp`里找到了对应实现，结合官方文档来看，是根据输入Tensor的rank逐步降rank并匹配相应优化实现，也就是通过模板特化匹配。在我们`gemm(mma, tCsA, tCsB, tCrC);`的例子里，三者一开始都是`(V, M, N)`的rank-3 Tensor：

```
(_1,_8,_8):(_0,_16,_128)
(_1,_8,_8):(_0,_16,_128)
(_1,_8,_8):(_0,81920,_16)
```

### 第一次dispatch  `(V,M,K) x (V,N,K) => (V,M,N)`

在为每个SMEM的Tensor构造RMEM上的fragment后，三者形状是

```
(_1,_8)
(_1,_8)
(_1,_8,_8)
```

### 第二次dispatch `(V,M) x (V,N) => (V,M,N)`

#### 蛇形曲线优化

采用了Serpentine Path（Snake-like path 蛇形线）来计算，对于64-bit的数据，每行反转方向，对于我们目前的32-bit数据，每两行反转方向。以 `M=8, N=8` 为例，外层循环 `m` 取 `0, 2, 4, 6`，每一轮内层 `n=0..7` 计算出一个 `ns` 序列，同时用于 `m` 和 `m+1` 两行。

`ns` 随外层 `m` 的变化

| 外层 m | 实际行 r | `m & 2` | `ns` 序列（n=0→7）     |
| ------ | -------- | ------- | ---------------------- |
| 0      | 0, 1     | 0（假） | 0, 1, 2, 3, 4, 5, 6, 7 |
| 2      | 2, 3     | 2（真） | 7, 6, 5, 4, 3, 2, 1, 0 |
| 4      | 4, 5     | 0（假） | 0, 1, 2, 3, 4, 5, 6, 7 |
| 6      | 6, 7     | 2（真） | 7, 6, 5, 4, 3, 2, 1, 0 |

逐行、逐列的完整 `ns` 值

| 行   | n=0  | n=1  | n=2  | n=3  | n=4  | n=5  | n=6  | n=7  |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 0    | 0    | 1    | 2    | 3    | 4    | 5    | 6    | 7    |
| 1    | 0    | 1    | 2    | 3    | 4    | 5    | 6    | 7    |
| 2    | 7    | 6    | 5    | 4    | 3    | 2    | 1    | 0    |
| 3    | 7    | 6    | 5    | 4    | 3    | 2    | 1    | 0    |
| 4    | 0    | 1    | 2    | 3    | 4    | 5    | 6    | 7    |
| 5    | 0    | 1    | 2    | 3    | 4    | 5    | 6    | 7    |
| 6    | 7    | 6    | 5    | 4    | 3    | 2    | 1    | 0    |
| 7    | 7    | 6    | 5    | 4    | 3    | 2    | 1    | 0    |

规律总结：

- **m 为 4 的倍数（m=0,4）**：`ns` 正向遍历 `0→7`，且连续两行（行对）的 `ns` 顺序完全相同。
- **m ≡ 2 mod 4（m=2,6）**：`ns` 反向遍历 `7→0`，连续两行的 `ns` 顺序也完全相同。
- 整体模式是**每两行为一组，组内同向，组间交替反转**，形成一种“成对蛇形”扫描。

#### 如何实现寄存器重用

在GPU上，RF被分为多个bank，以便计算单元能每个cycle读取多个operand。当计算压力上升时，可能会遇到多个operand在同一bank的情况，此时发生register bank conflict，拖延整体执行速度。此外对寄存器的访问占用很多功耗和时间。

编译器ptxas会借助**Data Dependency Graphs (DDG)** 和 **Liveness Analysis**来做决策，也就是在NCU中经常可见的分析，它会在合适的时机在SASS加上`.reuse`变量标签，来减轻RF压力和bank conflict。

如果按照每行从左到右的方式计算`tCrC`，换行时没有寄存器可以复用。那此时时计算单元从寄存器获取数据突然增多，RF的read request更多，更易触发register bank conflicts。蛇形曲线使得换行时也能复用寄存器，read request压力均衡。

对于我们的32-bit数据，相同大小的tile需要更多指令来执行，RF的read request更多，使用“kinked”蛇形曲线可以增加复用程度。

#### 实际情况：NCU打开SASS未看到.reuse

是否是方式错误？



## 首次profiling

### 设置

实验在单张A10 PCIe进行，A10的cuda core性能和时钟频率理论上比A100更高，整个memory hierarchy又比A100受限，很适合做优化练习

NCU使用了`--set full`

### 首先看向SOL & Roofline

|                   | cublas  | sgemm_2.cu |
| ----------------- | ------- | ---------- |
| TFLOPS            | 14.10   | 10.84      |
| SM Throughtput    | 74.74 % | 90.27 %    |
| Memory Throughput | 62.08 % | 90.27 %    |

A10 PCIe FP32理论峰值31.24 TFLOPS， 但由于128个cuda核心中，64个用于FP32计算，64个用于INT32和FP32计算，在计算sgemm时即便是cublas也不能使所有核心做FP32运算，再加上时钟频率和功率影响，即便是cublas，最终也只得到了峰值性能的一半不到。

`sgemm_2.cu`看似有很高的SM吞吐，似乎已经是优化到极限的compute-bound算子，但仅SM和Memory吞吐量相同就足以让人警惕。看向详细的Compute Throughput Breakdown，发现`SM: Inst Excuted Pipe Lsu`为90.27%，而`SM: Pipe FMA Active Cycles`只有43.91%，SM做的大部分操作是内存搬运。再比较FLOPS，`sgemm_2.cu`的计算效率更低。显然是SM花大量时间执行内存搬运指令，让GPU看起来很繁忙，却又不做计算。

## 定位问题的哲学

加上这一个section的原因是观察到了分析的重要。因为钱不够，无奈从A100 SXM4 40GB切换到A10 PCIe 24GB，却发现同样的代码在不同配置的硬件上profile得到了不同的结果，甚至于发现了compute-bound和memory-bound之外还有其他的问题也会对kernel造成主要影响。A100表现为compute-bound，SM throughput达到91.9%，Memory throughput在52%左右，看起来是个优秀的kernel，与cublas sgemm的96.5% SM throughput相差5%。

可以窥见现实中算子优化会遇到各种问题，仅堆砌优化技巧不能解决所有问题，也不是所有问题都有优化技巧可堆砌。所以牵扯到一个重要问题，怎么**定位算子中的主要问题**？其实也是大局观部分想要回答的问题。借助AI看懂算子中的优化技巧不难，**怎么能建立一套分析系统呢**？例如这里分析出LSU占用过多，而同时有uncoalesced store和bank conflicts，先解决哪个是需要决策的。

---

跟着AI给的答案思考吧：GPU上的代码执行是一个巨大的流水线，总有一步是限制最大的。具体先看SOL和Roofline Chart，再关注warp stall reasons。

目前对GPU执行有大概的全景，从之前的内存指令占用过多开始分析吧。

## 首次profiling的详细分析

### 初步分析kernel

SGEMM这一问题有明确的分析：优化后的SGEMM每$O(N^2)$次内存访问应有$O(N^3)$次FMA计算，最繁忙的pipe应该是FMA。估算如果使用8*8的register tile，LDS和FFMA指令比例应该是`1:4`。

### 宏观分析

从SOL看起，理解report中活跃的每一个细节，这里主要理解各指标的throughput实际上和哪些硬件相关，硬件执行细节在这里尤为重要。

#### Roofline

落在Compute-Bound区，总体数据利用率不错

#### Compute Throughput Breakdown

从高到低看。百分比有两种含义，per cycle的实际和理想最大执行指令数比例，一般称为SOL throughput；该pipe活跃cycle和总cycle比例。

1. `SM: Inst Executed Pipe Lsu [%]`：LSU Pipe的SOL throughput。硬件已经被设计好每cycle执行多少指令，所以这个才能profile这个指标

   - 90.27%的throughput意味着SM花大量时间执行LD/ST指令

2. `SM: Issue Active [%]`：warp scheduler发射指令的cycle数和kernel总cycle数的比例。在NV GPU上，每个cycle最多可以issue一个指令，在Volta及之后的架构都移除了dual-issue设计。

3. `SM: Inst Executed [%]`：SM执行的SOL throughput。这个metric实际上也来源于warp scheduler，它关注的是单SM上的最大throughput，Volta及之后的SM有4个SMSP且single-issue，最大throughput则是每cycle 4个指令，可以借助通过公式反向计算NCU内置数据来验证：
   $$
   \text{Peak Instructions per Cycle} = \frac{\text{sm\_\_inst\_executed.avg.per\_cycle\_elapsed}}{\text{sm\_\_inst\_executed.avg.pct\_of\_peak\_sustained\_elapsed}} \times 100
   $$
   因为23都是关于warp scheduler的指标，经常可以观察到两个指标非常接近。揭示了这些信息（来自Gemini）：

   - 没有dual-issue（Instruction-Level Parallelism）：The SM is issuing exactly one instruction per active cycle and is not utilizing any dual-issue pipeline capabilities. 如果有的话，则会在一个cycle同时发射两条指令，增加`Inst Executed`，但Volta及之后的架构的single-issue特点使得两个metric经常一样。
   - 几乎没有指令replay：Every time the issue logic activates, the instruction successfully executes without needing to be re-issued (replayed) due to resource conflicts or cache misses. 如果有的话，则会重新发射指令，增加`Issue Active`，而不增加`Inst Executed`。
     - 在我的bank conflict实验中，确保没有uncoalesced global access，增加bank conflict量，确实使`SM: Issue Active [%] `和` SM: Inst Executed [%]`差距增加
   - 如果值不高<70%，则所有SMSP在等待某类指令执行

4. `SM: Pipe Fma Cycles Active [%]`：Fma pipe执行FMA的cycle数和kernel总cycle数的比例。这个pipe需要4个cycle执行完，但它的流水线设计使得它可以每cycle都开始执行一个FMA指令。

   - 43.88%的throughput意味着FMA有很多空闲时间

5. `SM: Mio Inst Issued [%]`：MIOC给MIO分发指令的繁忙程度，即MIOC的SOL throughput。第一，要了解每个SMSP上的dispatch unit，每条指令经过warp scheduler后会由它分发给对应执行unit，有些unit由SMSP独享，例如ALU、FMA、XU等，有些unit在SM上被共享，例如TC、LSU等。第二，MIO是对XU、LSU、TEX、IDC等的总称。当指令被dispatch到SM上共享的unit时，总会经过MIOC。

   - 30.53%的throughput说明MIOC给MIO分发指令的吞吐量并没有被占很多，但LSU却有很高的吞吐量，90.27%。也恰恰说明了只有LSU在繁忙，而其他的MIO器件没有产生瓶颈。

6. `SM: Pipe FmaHeavy Cycles Active [%]`：FmaHeavy Pipe的繁忙程度，即FmaHeavy执行的cycle数和kernel总cycle数的比例。在Ampere及之后的架构上，cuda core被分为两组：

   > GA102 Whitepaper:
   >
   > One datapath in each partition consists of 16 FP32 CUDA Cores capable of executing 16 FP32 operations per clock. Another datapath consists of both 16 FP32 CUDA Cores and 16 INT32 Cores, and is capable of executing either 16 FP32 operations OR 16 INT32 operations per clock. As a result of this new design, each GA10x SM partition is capable of executing either 32 FP32 operations per clock, or 16 FP32 and 16 INT32 operations per clock. All four SM partitions combined can execute 128 FP32 operations per clock, which is double the FP32 rate of the Turing SM, or 64 FP32 and 64 INT32 operations per clock.

   所以可推断，可以执行FP指令和INT指令的FmaHeavy Pipe对应那组灵活的cuda core，即硬件上的执行lanes；而FmaLite Pipe对应那组只执行FP的cuda core

   - 28.36%的活跃时间与Fma Pipe有差，说明有些指令只在FmaLite上执行，可以在compute-bound时深入分析这个

7. `SM: Mio2rf Writeback Active [%]`：MIO写回RF的cycle数和kernel总cycle数的比例。可以看[NCU文档里的示意图](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#id40)，数据在被MIO取回后需要被写入RF才能被计算单元使用（不考虑wgmma），这个写回interface的带宽和并行度很大，所以这一指标常常比`SM: Mio Inst Issued [%]`低。

8. `SM: Mio Pq Read Cycles Active [%]`：\# of cycles where MIOP PQ sent register operands to a pipeline。听起来MIO PQ是执行内存指令的buffer，内存指令开始执行的时候可以从这个buffer获取operand，operand会传递给LSU、TEX等（猜测）。

9. `SM: Mio Pq Write Cycles Active [%]`：\# of cycles where register operands from the register file were written to MIO PQ。听起来像是内存指令被阻塞时，后续内存指令对应operand会被预取到这个buffer，等到MIO空闲时自动执行，这样可以保证warp scheduler繁忙（猜测）。

   - 89两个指标都在3%以下，与lsuin高达90%形成对比，说明指令不多，但交给LSU、L1/TEX运行很慢。

10. `SM: Inst Executed Pipe Adu [%]`：Address Divergence Unit的SOL throughput。ADU负责branch和jump的地址计算，它也支持constant loads和block-level barrier instructions。

11. `SM: Inst Executed Pipe Uniform [%]`：Uniform Pipe的SOL throughput。Turing及之后的架构中，每个SM含有一个Uniform Datapath和一组Uniform Register File (URF)，他们都是独立的硬件，用于计算一个warp输入数据、输出结果和control-flow相同的指令，即**warp-uniform** execution，例如指针base计算、循环increment等。

12. `SM: Inst Executed Pipe Cbu Pred On Any [%]`：Convergence Barrier Unit在分支上predicate的SOL throughput。CBU本身负责 warp-level convergence, barrier, and branch instructions，即warp的分支和同步。在执行例如if-else的分支时，硬件先执行if再执行else，CBU负责把需要执行的线程predicate，在if-else执行后让所有线程同步。

13. `IDC: Request Cycles Active [%]`：IDC的活跃时间。constant cache从此经过。

14. `SM: Inst Executed Pipe Ipa [%]`：Interpolation Attribute Pipe的SOL throughput。IPA负责图形计算相关的，一般都是0。

15. `SM: Inst Executed Pipe Tex [%]`：Texture Pipe的SOL throughput。TEX负责图形计算相关的，一般都是0。

16. `SM: Inst Executed Pipe Xu [%]`：Transcendental and Data Type Conversion Unit的SOL throughput。它负责pecial functions such as sin, cos, and reciprocal square root. It is also responsible for int-to-float, and float-to-int type conversions.

17. `SM: Instruction Throughput Internal Activity [%]`：SM管理内部的SOL throughput。这一指标揭示了SM内部的overhead，相关操作有从I-Cache取指令、管理buffer、decoding、replay机制等

18. `SM: Memory Throughput Internal Activity [%]`：与上一条类似

19. `SM: Pipe Fp64 Cycles Active`

20. `SM: Pipe Fp64 Cycles Active`

#### Memory Throughput Breakdown

这里的百分比含义和compute差不多，一是硬件设计上限，比如一条硬件连接每cycle最多能传输多少request，

1. `L1: Lsuin Request [%]`：LSU给L1发送request的SOL throughput。首先看硬件过程：MIO将每个warp的内存访问指令生成request，传递给LSUIN或TEXIN，再由L1TEX取出sectors或cache lines，随后触发writeback。在Volta及之后的架构中，每条指令会对应生成**一个**request，每个request包含了**一整个warp**的访问需求。所以LSU和LSUIN是一条硬件链路的两端，他们被设计为有相同的最大throughput，该指标常常和`SM: Inst Executed Pipe Lsu [%]`相近或相同。

   在大范围uncoalesced实验中未观测到两者差距增加，可能两者相差有其他内部原因，也指出合并访问并不会使request与instruction的比例减小。

   - 90.21%

2. `L1: Data Pipe Lsu Wavefronts [%]`：Data-Stage读写wavefronts的SOL Throughput。理解该指标的关键是理解**wavefront**，与instruction与request`1:1`不同，一个wavefront是硬件pipeline每cycle能并行处理的最大工作量，其数量应该和request难度、本身处理能力相关。在tag stage和data stage，同一条request可以有不同的wavefronts。

   该metric包含了global、local、shared的内存访问（不考虑TEX）。L1TEX每cycle可以取出**最多128字节或4 sector**的数据。shared访问每cycle可以获取全部$32\text{bits}\times 32\text{banks}=128\text{bytes}$的数据，wavefronts正比于bank conflict增加。global/local的访问随请求sector数量增加，超出128bytes/cycle的带宽就需要更多wavefronts，在大范围uncoalesced实验中，`L1: Lsuin Request [%]`仅有1.38%，而该指标达到了46.46%。

   如果有分支，产生了predicated warp execution，那么wavefronts也会增加。

   - 50.10%的wavefronts，没有像LSU request或instruction那样用满硬件，说明有些访问在同一波wavefront被合并处理。50左右的数值也说明到这里就不是bottleneck了。

3. `L1: Lsu Writeback Active [%]`

4. `L2: T Sectors [%]`

5. `L2: Lts2xbar Cycles Active [%]`

6. `DRAM: Cycles Active [%]`

7. `DRAM: Dram Sectors [%]`

8. `L1: M Xbar211tex Read Sectors [%]`

9. `L1: Data Bank Reads [%]`

10. `L2: D Sectors [%]`

11. `L2: T Tag Requests [%]`

12. `L2: D Sectors Fill Device [%]`

13. `GPU: Compute Memory Access Throughput Internal Activity [%]`

14. `L2: Xbar2lts Cycles Active [%]`

15. `L1: Data Bank Writes [%]`

16. `L1: ML1tex2xbar Req Cycles Active [%]`

17. `L1: Texin Sm2tex Req Cycles Active [%]`

18. `L2: D Sectors Fill Sysmem [%]`

19. `GPU: Compute Memory Request Throughput Internal Activity [%]`

20. `L1: Data Pipe Tex Wavefronts [%]`

21. `L1: F Wavefronts [%]`

22. `L1: Tex Writeback Active [%]`

23. `L2: D Atomic Input Cycles Active [%]`

#### 小结

1. 找到最大瓶颈：由`L1: Lsuin Request [%]`$\approx$`SM: Inst Executed Pipe Lsu [%]`$\approx$ 90.21%可知，最繁忙的是根据instruction发送request的LSU，和接收request的LSUIN，可以推断瓶颈在于global/local/shared的访问指令
2. 回顾硬件模型：思考LSU的前后部分是否造成影响。前是warp scheduler和MIOC，他们没有很高的throughput，说明瓶颈有unit需要他们等待。后是L1/TEX，50.10%的`L1: Data Pipe Lsu Wavefronts [%]` 显示，最终取数据的data-stage吞吐量没被用满，说明取数据不需要排队等待，也能和高LSU throughput呼应。那我们可以根据硬件模型判断，是通过LSU的指令太多了。
3. 寻找更多呼应细节：`SM: Pipe Fma Cycles Active [%]`是43.88%，在SGEMM计算中它在等着被喂数据。

### 微观分析

先确认SOL分析中的细节

- Memory Workload Analysis：memory table里没有bank conflicts，global load的`Sectors/Req`是16，符合使用`u_int128_t`的向量化load，这部分没有未合并访问。但global store的`Sectors/Req`也是16，可能需要深入研究，但guided analysis的加速比例过小，暂且搁置。
- Warp State Statistics：最大的warp stall是MIO Throttle，说明有很多warp等着加入MIO instruction queue，略高的Short Scoreboard和LG Throttle也佐证了是shared访问而不是global/local访问作为瓶颈

再定位问题，走向指令和source分析

- Instruction Statistics：可见FFMA和LDS占比很多，且比例是恰好4:1，register tile的尺寸是8*8，符合全部使用32-bit LDS指令的instruction mix。这些信息说明瓶颈可能在LDS指令。
- Source Counters：执行最多的是FFMA，这是自然。点击跳转Warp Stall最多的指令，发现是LDS，位于CuTe的`copy_if()`，点击查看其all stall sampling，最多的MIO Throttle占比52.03%。

至此，我们可以确定LDS指令是造成LSU繁忙的原因。

### FIX

虽然nvcc编译时使LDS指令和FFMA指令interleave，但这样不足以掩盖LDS的慢。现在使用向量化的`LDS.128`，将总LSU requests降低为1/4。在CuTe中修改则是增加一个向量化的S2R copy，但要注意SMEM layout是否支持直接向量化呢？

首先有几个问题需要解答：

- 是哪一步的layout阻止了向量化？
- 即使都使用TiledCopy，GMEM$\rightarrow$SMEM和SMEM$\rightarrow$RMEM不同，前者是两段线程共有地址空间的映射，虽然可能有取离散数据的情况，但GMEM和SMEM都可以被看做是一个大数组，线程可以协作搬运再各取所需；后者是一段共有的地址空间到每个线程独有的寄存器的映射，每个线程搬运的数据就是它需要的数据。S2R TiledCopy设计的考量就变多了，要怎么考虑呢？
- 上述的TiledCopy中每个线程的分工和对应数据，即TV layout该怎么查看呢？

#### v1代码深入

深入`sgemm_2.cu`的source，`gemm(mma, tCsA, tCsB, tCrC)`最终lower到了`copy_if()`，此处predicate tensor的iterator不是指针，是一个只返回constexpr true的常量函数，所以这里的`if`分支在编译时被优化，没有运行时overhead。要注意与`if (constexpr)`和`if constexpr ()`不同，前者的优化发生在IR的优化，后者与模板实例化一起发生。那么这里就是简单的`dst(i) = src(i)`。

在这份代码里，没有设计SMEM$\rightarrow$RMEM的TiledCopy，使用默认的`copy()`。CuTe会使用`AutoFilter(AutoVectorizingCopyWithAssumedAlignment<Maxbits>)`来尝试向量化LDST，它会

- 如果使用静态的shape和stride，CuTe会尝试根据当前layout和type，找到最大的向量化访问指令，当然也有可能无法向量化
- 如果只有静态的shape，或动态的shape和stride，都不会尝试向量化。此时可能需要用到`copy_aligned()`

v1的代码没有任何SMEM bank conflict，这是深入的关键。可以分成两个场景来看，LDGSTS和MMA。

- LDGSTS中，256个线程的layout是`(_32,_8):(_1,_32)`，每个线程负责`(_4,_1):(_1,_0)`layout的`uint128_t`atom，所以一个warp刚好存取到contiguous的512byte，在GMEM和SMEM都是向量化的。

- MMA中，256个线程的layout是`(_1,_16,_16,_1):(_0,_1,_16,_0)`（mode-0和mode-3可以忽略），其中一个warp就是$16 \times 2$个或两行contiguous线程，每个线程会负责一个atom的计算，即1个element。在一个$128\times 128$的tile里，partition时这个256的方块会自动重复以填满tile，那么在每个线程的register tile里，每个相邻元素在tile里相距16元素。由此可知，bank conflict不会发生在$16\times 2$线程的一行或一列，它们搬运数据到register tile时
  - lane id连续的16个线程会访问16个连续bank，或同一个bank的同一地址，
  - lane id相差16的两个线程会访问2个连续bank，或同一个bank的同一地址，
  
  bank的并行和broadcast可以做到bank conflict free。

但是这样也使SMEM$\rightarrow$RMEM的向量化不可能，用`max_common_vec`可以证明这一点

#### v2使用了PermutationMNK

目前的问题是每个thread需要的$8\times8$ register tile的元素相隔太远，使用`make_tiled_mma()`中的`PermutationMNK`可以使`AtomLayoutMNK`在任意mode上重复甚至interleave。下面一系列图有助于说明这一点。

首先我们引入`PermutationMNK`，看它是如何影响`AtomLayoutMNK`的

```cpp
TiledMMA mma1 = make_tiled_mma(MMA_Atom<UniversalFMA<float>>{},
                               Layout<Shape<_16, _16, _1>>{}); // AtomLayoutMNK
// 加入PermutationMNK
TiledMMA mma2 = make_tiled_mma(MMA_Atom<UniversalFMA<float>>{},
                               Layout<Shape<_16, _16, _1>>{}, // AtomLayoutMNK
                               Tile<_16, _16, _1>{});		 // PermutationMNK
// 结果显示mma1与mma2一样，因为Tile就是AtomLayoutMNK，不在任意mode上重复或interleave
```

![](C:\Users\jackw\Desktop\cuda-docs-notes\images\tiledmma1.png)

再尝试在一个mode上重复

```cpp
TiledMMA mma3 = make_tiled_mma(MMA_Atom<UniversalFMA<float>>{},
                                 Layout<Shape<_16, _16, _1>>{},
                                 Tile<Layout<Shape<_16, _2>>, 
                                      _16, 
                                      _1>{});
// M mode应该被重复了2次
```

![](C:\Users\jackw\Desktop\cuda-docs-notes\images\tiledmma3.jpg)

接下来是最重要的一步，让同一线程的元素相邻

```cpp
TiledMMA mma4 = make_tiled_mma(MMA_Atom<UniversalFMA<float>>{},
                                 Layout<Shape<_16, _16, _1>>{},
                                 Tile<Layout<Shape<_16, _2>, Stride<_2, _1>>, 
                                      _16, 
                                      _1>{});
// 在shape上，M mode由_16*_2个元素组成，在默认设定的Stride<_1, _16>中，MMA按照每16个元素的方式重复。现在用Stride<_2, _1>，则是让MMA先重复两个元素。改变的是TV layout，只是改变了线程的负责对象，计算时的结果依然正确
```

![](C:\Users\jackw\Desktop\cuda-docs-notes\images\tiledmma4.jpg)

我们可以继续在N mode上做同样的事，使得整个tile都合并起来

```cpp
TiledMMA mma5 = make_tiled_mma(MMA_Atom<UniversalFMA<float>>{},
                               Layout<Shape<_16, _16, _1>>{},
                               Tile<Layout<Shape<_16, _2>, Stride<_2, _1>>,
                                    Layout<Shape<_16, _2>, Stride<_2, _1>>,
                                    _1>{});
```

![](C:\Users\jackw\Desktop\cuda-docs-notes\images\tiledmma5.jpg)

此时我们可以把`_2`改成`_8`，也就是该SGEMM的register tile。要记住修改的只是TV layout，是每个线程的计算任务，不会影响计算结果的正确。

#### v2的下位替代

使用`PermutationMNK`改变每个线程的计算任务是很好的解法，几乎没有overhead产生。那还有别的方法能解决问题吗？也许有，CuTe的灵活性和控制粒度十分优秀，我们也许可以通过调整SMEM layout来配合TiledMMA，但此时GMEM$\rightarrow$SMEM的copy就无法向量化。

再深入一点，`UniversalFMA`这个Atom只给每个thread一个元素计算，我们可以尝试放弃它，自己定义一个Atom封装4个`fmaf`指令，这样向量化就蕴含在Atom中了，打印TiledMMA得到的结果应和使用`PermutationMNK`类似，而且这个方法不如`PermutationMNK`灵活。（貌似也不行，四条指令没法同时执行，实际上也造成uncoalesced access，——第二次profiling时的思考）

### 总结

从10TLFOPS上升到了12TFLOPS，相较于cublas的14TFLOPS仍有差距。

- TiledCopy和TiledMMA要切换到warp视角来看，很多思考点都是从warp出发的，比如bank conflicts只在一个warp内发生、Tensor Core的wmma等。

- 流程非常重要，CuTe的编译流程、硬件执行指令流程、cuda kernel执行流程等都至关重要，profiling的前提是了解流程。

- 对于新手，在经验中学习。

## v2 profiling的详细分析

一份全新的报告到手，回顾第一次分析和AI总结经验，按以下几步分析：

1. SOL分析大方向：compute-bound、memory-bound、latency-bound等，再结合问题具体定位到某个subsystem。如果有baseline可以对照着看，也可以比较。
2. 辨清Guided Analysis：NCU指出几个大问题，结合具体metrics分析他们的影响，是否为bottleneck
3. 深入subsystem：常和上一步结合着看，回顾硬件的具体执行过程
4. 回到Source定位问题

接下来总结AI的分析过程和我的思考。

### SOL分析大方向

首先，Memory Throughput是70.52%，Compute Throughput是55.66%，两者相差10%以上（在Guided Analysis中，这被指出是一个Key Performance Indicator），说明kernel目前忙于内存搬运而非计算。Compute Breakdown中`SM: Pipe Fma Cycles Active [%]`是50.71%，也佐证了目标的计算类型不够繁忙。

再看几个子系统，`L1/TEX Cache Throughput`是72.82%，`L2 Cache Throughput [%]是`21.66%，`DRAM Throughput [%]`是19.53%，很容易判断是L1/TEX subsystem最繁忙。在SGEMM问题中，结合当前代码看，最可能的就是shared memory相关的问题。

### 辨清Guided Analysis & 深入subsystem

结合NCU的Guided Analysis和上述SOL分析，有如下主要问题

- L1TEX Global Store Access Pattern：向GMEM写操作的合并访问不够，平均每sector访问只有4 byte被利用
  - Est. Speedup: 61.70%
  - 出现频率：每个`axpby` epilogue
  - 优先程度：虽然加速比很高，但NCU rule-based的Guided Analysis忽略了具体执行情况。`DRAM Throughput [%]`仅有19.53%，STG指令占比极低，都说明这个问题占比极低。
- Uncoalesced Shared Accesses：有超出预期的wavefronts，主要是由bank conflict引起
  - Est. Speedup: 36.89%
  - 出现频率：每次s2r copy
  - 优先程度：感觉较高，需具体分析
- Theoretical Occupancy：在该SGEMM中，每线程块使用$102 \times 256 = 26112$个寄存器，受每个SM64K寄存器限制，只能同时运行两个线程块
  - Est. Speedup: 29.48%
  - 出现频率：每次kernel call
  - 优先程度：受寄存器限制，那么说明与计算、内存操作密不可分，这是程序结构性的问题，没法先优化

- 还有更多……

#### 查找证据

既然是内存主导的问题，主要来看Memory Workload Analysis和Warp State Statistics。先看到Memory Workload Analysis，有三大项对于揭示我们的问题很关键：

- `Mem Busy [%]`：Cache和DRAM的内部活动繁忙程度。该SGEMM为70.52%。
- `Max Bandiwidth [%]`：SM、Cache和DRAM间互联的Throughput。该SGEMM为41.03%。
- `Mem Pipe Busy [%]`：SM发射内存指令的Throughput。该SGEMM为28.51%。

三个指标明确揭示了问题出现的阶段。结合着看，`Mem Busy [%]`高而`Mem Pipe Busy [%]`低，说明L1/TEX在忙着用很多wavefronts执行少量指令，这是bank conflict作为显著问题的一大证据。另外`L1/TEX Hit Rate [%]`是较低的8.30%与我们的问题关系不大，因为GMEM Access在该SGEMM中相对较少，即便hit rate低也不会造成L1/TEX的过度繁忙。

接着看详细的Shared Memory Table（一般只有确定了大致问题才会转向Table，不要先看Table），只有load操作贡献了全部419430400次bank conflicts，与shared load一共1048540356个wavefronts相比，占比40%。这里有一个疑点，request与ideal wavefronts相比大约是3，但一个warp执行`LDS.128`指令load的是512 bytes，也就是4 wavefronts，留给后期定位具体代码时分析，首先定位好瓶颈。

再看Warp State Statistics，在显著的stall reason中，有几项值得关注

- `Wait`和`Dispatch Stall`都比较高。表面上看，fix latency操作相关的`Wait`指向计算单元拥堵，`Dispatch Stall`指向计算单元拥堵或寄存器数据读取压力，应该说明FMA pipe饱和了，但50%左右的throughput和约为0的`Math Pipe Throttle`说明FMA pipe不可能是因为Throughput过高。那等会去SASS中找原因。
- `Barrier`略高，这是由于Occupancy较低而明显增加的stall。每个threadblock或者说tile执行时有两个`__syncthreads()`，register限制每个SM只有2个threadblock，`__syncthreads()`会让整个threadblock等待，却没有足够的threadblock能切换运行来掩盖等待，这就造成了高`Barrier`
- `MIO Throttle`和`Short Scoreboard`明显指向bank conflicts
- `Long Scoreboard`几乎为0，这说明global-memory latency完全被register double-buffer/prefetch隐藏了。该SGEMM并没有受到DRAM或gmem-latency-bound的限制，DRAM 仅为 19.5% 也证实了这一点。
- `LG Throttle`出现在epilogue写回GMEM时，`MIO Throttle`也和这里有关，有excessive sectors。

无需太关注的：

- `Selected`：始终为1
- `Not Selected`：23%是个不错的迹象，一个warp已经就绪，但scheduler issue了另一个warp。在IPC（Instruction Per Cycle）已经达到 2.30/4（通过57% issue rate计算, 即`sm__inst_issued`）的情况下，这个stall仅仅说明有足够多符合条件的工作在争夺issue slot。`not_selected`不需要针对性优化；当消除了真正的stalls后，它的占比自然会减少。

### 定位并深入问题

#### bank conflicts

在之前的分析中，bank conflicts与shared load总wavefronts相比达到了40%，但估计的`request:ideal wavefronts`是`1:4`因为每次warp的`LDS.128`指令需要load 512 bytes，总计4 wavefronts。这里`1:3`很明显有不同的N-way conflicts混合在一起。简单查看以下SASS，LDS指令只有128版本，那么可以怀疑是Access Pattern不够好，打印`LDS.128`部分的SASS如下：

```
ncu --import sgemm_opt80_a10_v2.ncu-rep -k "regex:sgemm_opt80_nt_v2" --page source --print-source sass 2>/dev/null \
  --metrics derived__memory_l1_wavefronts_shared_excessive,derived__memory_l1_conflicts_shared_nway \
  | grep -iE "LDS|Wav|Metric" | grep -vE "^\s*$" | head -50
Address                  Source                                                 L1 Con L1 Wav  
0x7fcedb3a8800           LDS.128 R68, [R66]                                     8      26214400
0x7fcedb3a8850           LDS.128 R72, [R65]                                     2      0       
0x7fcedb3a8890           LDS.128 R76, [R65+0x10]                                2      0       
0x7fcedb3a88c0           LDS.128 R80, [R66+0x10]                                8      26214400
0x7fcedb3a8b00           LDS.128 R68, [R66+0x200]                               8      26214400
0x7fcedb3a8c90           LDS.128 R76, [R65+0x200]                               2      0       
0x7fcedb3a8d10           LDS.128 R72, [R65+0x210]                               2      0       
0x7fcedb3a8d20           LDS.128 R80, [R66+0x210]                               8      26214400
0x7fcedb3a8f40           LDS.128 R68, [R66+0x400]                               8      26214400
0x7fcedb3a90d0           LDS.128 R72, [R65+0x400]                               2      0       
0x7fcedb3a9150           LDS.128 R76, [R65+0x410]                               2      0       
0x7fcedb3a9160           LDS.128 R80, [R66+0x410]                               8      26214400
0x7fcedb3a9380           LDS.128 R68, [R66+0x600]                               8      26214400
0x7fcedb3a9510           LDS.128 R76, [R65+0x600]                               2      0       
0x7fcedb3a9590           LDS.128 R72, [R65+0x610]                               2      0       
0x7fcedb3a95a0           LDS.128 R80, [R66+0x610]                               8      26214400
0x7fcedb3a97c0           LDS.128 R68, [R66+0x800]                               8      26214400
0x7fcedb3a9950           LDS.128 R72, [R65+0x800]                               2      0       
0x7fcedb3a99d0           LDS.128 R76, [R65+0x810]                               2      0       
0x7fcedb3a99e0           LDS.128 R80, [R66+0x810]                               8      26214400
0x7fcedb3a9c00           LDS.128 R68, [R66+0xa00]                               8      26214400
0x7fcedb3a9d90           LDS.128 R76, [R65+0xa00]                               2      0       
0x7fcedb3a9e10           LDS.128 R72, [R65+0xa10]                               2      0       
0x7fcedb3a9e20           LDS.128 R80, [R66+0xa10]                               8      26214400
0x7fcedb3aa040           LDS.128 R68, [R66+0xc00]                               8      26214400
0x7fcedb3aa1d0           LDS.128 R72, [R65+0xc00]                               2      0       
0x7fcedb3aa250           LDS.128 R76, [R65+0xc10]                               2      0       
0x7fcedb3aa260           LDS.128 R80, [R66+0xc10]                               8      26214400
0x7fcedb3aa480           LDS.128 R68, [R66+0xe00]                               8      26214400
0x7fcedb3aa610           LDS.128 R76, [R65+0xe00]                               2      0       
0x7fcedb3aa690           LDS.128 R72, [R65+0xe10]                               2      0       
0x7fcedb3aa6a0           LDS.128 R80, [R66+0xe10]                               8      26214400
```

这里excessive wavefronts总共有$26,214,400 \times 16 = 419,430,400$，即所有shared load bank conflicts。而$26,214,400 \text{ excessive} ÷ (12,800 \text{ warps} × 512 \text{ k-tiles}) = \text{exactly } 4.0 \text{ wavefronts}$，正好对应了8-way bank conflicts和ideal wavefronts为4。进一步计算，$\frac{2+4}{2+8}=60\%$，正好对应了40%的bank conflicts。

N-way conflicts以8-2-2-8周期形式出现，且不同conflicts对应地址寄存器`R65`和`R66`，可以推断分别SMEM中A和B的地址分别存在`R65`和`R66`中，在source和SASS对照中可以发现，A tile的地址对应`R66`。再来看CuTe layout设计：

- `sA`的layout是`(_128,_8):(_1,_128)`，即m-major。`PermutationMNK`使每个线程处理连续的8*8 register tile。每次`LDS.128`指令取前4个float或后4个float，每个线程取的数据相隔4个float，可以推断只有一半bank活跃。
- `sB`的layout是`(_128,_8):(_1,_128)`，即n-major。与A tile不同的是，数据contiguous的mode，在MMA中对应mode线程的warp lane id并不是连续的。具体来说，一个warp前16个线程和后16个线程分别从两个地址取4个float，这两个地址相隔4个float，这个pattern使SMEM有了很大的broadcast空间。

具体broadcast原理还不太清晰，目前有几点可以确定：

- `LDS.128`的访问在硬件上会分为4个quarter-warp执行，以配合128 B的最大带宽
- 理想的`LDS.128`需要4个wavefronts执行

最终问题定位到A的s2r copy。

#### stall reasons

在之前的分析中，`Stall Wait`和`Stall Dispatch Stall`较高，FMA pipe Throughput和`Math Pipe Throttle`都不高，说明瓶颈不是堵塞的FMA pipe。深入source分析，发现

- `Stall Wait`的较高值集中在几条`LDS.128`指令上，而不是预期的`FFMA`指令，而且常常`LDS.128`指令紧跟一条`FFMA`指令
- `Stall Dispatch Stall`的较高值集中在上述`LDS.128`指令后第一条`FFMA`指令

首先，`Stall Wait`等待的是fix latency，而load类指令一般应该归于scoreboard类stall，说明stall的不是指令本身，而是指令所需的寄存器。简单观察上面打印出的`LDS.128`指令，它们使用的寄存器均为`R68/R72/R76/R80`之一，这些寄存器都在`LDS.128`指令前被`FFMA`指令调用。在LDS覆写一个寄存器之前，正在使用该寄存器的FFMA必须已经取走operand到pipe里，由于实际上等待的是FMA pipe释放寄存器，所以硬件上表现为`Stall Wait`。这不是数据依赖，而是register-name coupling。编译器为了节约使用寄存器把LDS指令schedule为prefetch，充分ILP，却不料`FFMA`指令取operand的过程过于卡顿。

`Stall Dispatch Stall`指operand准备好了，但dispatch unit不能发送。看一段SASS：

```text
Address         Instruction                    Wait    Dispatch_Stall
---------------------------------------------------------------------
0x7fcedb3a8800  LDS.128 R68, [R66]             235     1307  
0x7fcedb3a8850  LDS.128 R72, [R65]             0       421   
0x7fcedb3a8890  LDS.128 R76, [R65+0x10]        214     1779  
0x7fcedb3a88c0  LDS.128 R80, [R66+0x10]        924     3013  
0x7fcedb3a88f0  FFMA.FTZ R64, R68, R72, R64    0       433   
0x7fcedb3a8b00  LDS.128 R68, [R66+0x200]       0       656   
0x7fcedb3a8b10  FFMA.FTZ R31, R81, R72, R31    0       125   
0x7fcedb3a8c90  LDS.128 R76, [R65+0x200]       0       236   
0x7fcedb3a8ca0  FFMA.FTZ R7, R83, R75, R7      0       145   
0x7fcedb3a8d10  LDS.128 R72, [R65+0x210]       390     232   
0x7fcedb3a8d20  LDS.128 R80, [R66+0x210]       1936    1500  
0x7fcedb3a8d30  FFMA.FTZ R64, R68, R76, R64    0       5138  
0x7fcedb3a8f40  LDS.128 R68, [R66+0x400]       0       431   
0x7fcedb3a8f50  FFMA.FTZ R31, R81, R76, R31    0       142
```

最高值对应`0x8d30`的`FFMA`指令，它位于几乎连续的几个LDS之后，可以看到连续的LDS也会有`Stall Dispatch Stall`的升高。具体的原理不深究，总体来说是活跃的warp不够，以至于dispatch需要等待，而不是通过别的warp来掩盖。

这两者显著的原因都是Occupancy不足，不像其他时候scheduler可以正常掩盖。

### FIX

bank conflict可以通过调整`PermutationMNK`改善。

`Stall Wait`和`Stall Dispatch Stall`较高，但Occupancy不便再提高，cublas的做法是增加ILP，另外增加了thread block数但不改变Occupancy，增加一点异步性让硬件schedule更灵活。

`Stall Barrier`可能需要使用`cp.async`流水线来减少barrier，减少整个thread block的停止。

## v3 profiling的详细分析

### SOL分析大方向

Compute (SM) Throughput占到了最大的56.97%，没有任何SOL指标超过**60%**，目前没有任何可以明显定位的瓶颈unit。正如SOL Guided Analysis提到的，这很可能是latency-bound的体现，但也有小可能是launch不足（thread block过少，thread过少）、tail effect和instruction issue bound（比如过多unrolled循环堵塞issue pipeline，和第一次profile类似）等等。

> This workload exhibits low compute throughput and memory bandwidth utilization relative to peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60% typically indicate latency issues.

### 深入

接下来看到Occupancy、Scheduler Statistics和Warp State Statistics，当没有具体unit的问题，尝试从这几个sections确定真正的问题。

Occupancy 33%，thread block受寄存器数量限制（之前算过），而寄存器主要用于LDGSTS、A/B/C thread tile和epilogue等。

再看Scheduler Statistics，No eligible达到了41.11%，印证了低Occupancy的影响。

再看Warp State Statistics来了解No eligible warp时warp都在等什么。依然是`Wait`和`Dispatch Stall`为主，`barrier`为辅。与第二次profiling比较：

- `MIO Throttle`、`Short Scoreboard`和`LG Throttle`进一步降低，值很低，说明bank conflict的改进有效
- `Wait`、`Dispatch Stall`和`barrier`都有极小幅度提高，说明bank conflict的改进并没有解决主要问题，甚至可能减少了L1/TEX运行时间故减少了latency hide（猜测）
- 加入register multi-stage/prefetch效果不大

不过依旧，`Wait`、`Dispatch Stall`问题很难分析定位，但优化目的只有一个：让为数不多的warp有更多指令可以发射。而barrier问题则对应multi-stage方案。

总的来说，目前问题构成：egister pressure (102/thread) → occupancy capped at 33% (2 CTAs/SM) → only ~4 warps/scheduler → not enough warps to hide the **FFMA/LDS dependency latency (wait 1.59)** and the **single-buffered `__syncthreads()` (barrier 0.90)**.

### FIX

两条路径：

1. **SMEM double-buffer / multi-stage the global→shared path.** 保持了occupancy，想办法在ILP上做文章。提前发射`cp.async`或LDG，Stall-On-Use会保证warp有更多事可做，也符合充分利用所有unit的优化哲学（目前每threadblock只使用4096B，共8192B SMEM）
2. **Cut registers below 85/thread** (from 102) → fits a 3rd CTA/SM → 50% occupancy → more warps to hide the `wait` latency. 尝试增加occupancy，或许用`cp.async`代替LDG+STS可以减少寄存器使用

## 排流水线

流水线，又可以看作是增加buffer来掩盖延迟。给需要掩盖延迟的一步加上buffer，做预取，计算第$i$步的时候取$i+1$步，即计算掩盖了取数据的延迟。更深一步讲，是增加了一块数据producer与consumer之间的sceduling distance。排流水线大概这么做：

1. 列出从GMEM到RMEM计算的所有指令步骤
2. 给需要掩盖延迟的步骤**终点**加上buffer
3. 三个阶段：
   - prologue 预填充buffer
   - steady-state 流水线rotation
   - epilogue 一般是处理buffer中剩余tile

由于CUDA的设计和顺序执行的架构，很大一部分scheduling都交给了软件，所以流水线应用很广。但终究流水线只是优化手段，具体哪里加buffer，加几层buffer需要具体分析，而不是无脑加就有效。

## 失败的v4

### 失败经历

想当然的根据v3的分析加上了multi-stage和加大thread tile，遇到了两次意外：

- 想当然认为LDG latency更长，STS/LDS latency较短，给LDG加了在register中加了几层buffer，而在SMEM中少加了几层buffer，却发现register使用runtime indexing会直接spill到local memory，反而降低了效率，流水线的rotating无法实现，或者说很难实现。放弃了LDG register buffer再具体看warp stalls，v3中实际上long scoreboard已经几乎被编译器掩盖，只有short scoreboard从SMEM buffer中有所收益，但是非常少。

- 需要为sm86编译才能使用两个16 FP32 lanes，所以TFLOPS一直很低。调整编译选项后，所有指标**大幅优化**。**在所有优化前，检查编译和正确性很重要**。引入额外16条FP32 lane解决了大部分warp stall，猜想是使用全部的FP32 lane匹配上了设计的register带宽。

## v3a重新profile分析

重新编译后，比较v3和v3a，发现引入register double buffer性能完全相同，说明register double buffer引入的scheduling可能已经被编译器的自动overlap实现了，于是不引入不必要的优化，基于v3a走下去。

### SOL分析大方向

Compute Throughput 76.17%，Memory Throughput 61.95%，内存访问没有明显问题，我们需要详细判断目前的FMA pipe是否在尽全力计算（也就是compute-bound）。

Compute-bound的特征是

- math pipe或issue slot busy向100%趋近
- math pipe几乎每cycle都能接受一个新指令，故warp scheduler也几乎不会停下，因为一直有pipe和指令准备好了
- math pipe过满，准备好执行math的warp也只能stall，引起**`Math Pipe Throttle` stall**（math pipe的back pressure）

而latency-bound也会在Compute Throughput > 60%时出现，它表现为math pipe或SM仍有提升空间，比如目前的76.17%，仍有24%左右的throughput没被任何pipe占用。在某些cycle，warp在等待某些执行（scoreboard或barrier等），而scheduler找不到任何eligible warp。具体看

- No Eligible cycles是否向0趋近
- 这些No Eligible cycles是否有较大部分被scoreboard或barrier等解释，这就要看到warp stall里的详细解释了

在目前的情况下，只看SOL不能找到具体的主要瓶颈了，只能判断出不属于memory subsystem的问题了，而是compute-side problems

### 深入

由上可知，目前instruction mix是理想状态，我们要从cycle和stall入手，首先scheduler statistics是大门。

**No Eligible** 有21%，warp stall仍有降低空间。Eligible warps/scheduler = 2.07，Active = 3.94.不是一对足够优秀的数据，平均有接近一半的warp eligible。

warp stall中，`Math Pipe Throttle`没有出现在前列 $\rightarrow$ latency-bound

| Stall                         | cyc/inst           | meaning                                   |
| ----------------------------- | ------------------ | ----------------------------------------- |
| not_selected                  | 1.62               | contention (healthy — parallelism exists) |
| **short_scoreboard**          | **1.00**           | **waiting on LDS/SMEM results**           |
| barrier                       | 0.50               | `__syncthreads` (STS→barrier)             |
| **mio_throttle**              | **0.34**           | **MIO/LDS pipe queue full**               |
| dispatch / lg_throttle / wait | 0.18 / 0.15 / 0.14 | minor                                     |

照常`selected`/`not_selected`排除在外，剩下的warp stall指向**short_scoreboard + mio_throttle = 1.34 cyc/inst**，需要深挖SMEM和register之间的交互，另外还有`barrier` 0.50，这和STS周围的两层`__syncthreads()`有关。

另外说到latency-bound，occupancy也是需要看的，它揭示了latency能否被TLP掩盖。像目前有较大的accumulator在registers，occupancy只有33.33%，放弃thread tile的局部性也不可能收益很高，所以我们需要更多转向ILP。

接下来到SASS source中去看具体的warp stall分布，`short scoreboard` 主要在FFMA上，这是SMEM$\rightarrow$register中的consumer。而这些FFMA不是随机的，看寄存器可知，是第一个使用新LDS结果的FFMA，剩下的62条FFMA几乎没有`short scoreboard`，说明LDS的latency没有被完全掩盖。

`Mio Throttle`由相对来说较多的LDG和LDS导致，它们共用MIO pipe，但33%的Occupancy不足以相互掩盖。

`barrier`基本只在两条指令，the `STS.128` writing the staging tile（67.64%），和the first `LDS.128` after `BAR.SYNC`（32.36%）。简单的store→`__syncthreads`→load顺序步骤在每个k-tile都会引入`barrier` stall，所有warp都将在`__syncthreads()`等待才能开始下一个FFMA batch。

### FIX

有latency需要掩盖，通过TLP还是ILP呢？

## TLP or ILP latency hiding

就latency而言，每条指令的latency有两个源头：

- 来自前置指令：有dependency的input指令，相邻很近的同类指令抢占pipe形成back pressure，WAR寄存器占用，等等。
- 来自同时指令或指令本身：issue slot或dispatch unit过满，等等

可见latency成因较为复杂，在着手解决之前，先判断是否需要hide latency解决，还是throughput问题，比如：

- `MIO Throttle`和LSU吞吐量很高，可能是MIO吞吐量用满或指令排队用满buffer的压力，用向量化指令也许就能解决
- `Long Scoreboard`和饱和的memory bandwidth，可能和access pattern有关

当各类throughput都不高，也就是没有明显瓶颈时，会倾向于latency hiding考虑

### TLP

TLP高或低一般在launch时就已经确定，thread block受registers，shared memory，SM warp count等资源限制，warp数量也就受了限制，所以提高TLP可能面临着资源用量的trade-off。有几种情况可以很大程度倾向于TLP：

- 每个warp的ILP受长依赖链限制
- CTA level sync过多，增加warp也无济于事
- 当取得更多的TLP不会有太多trade-off时，例如
  - 没有register spill
  - 不会造成更多SMEM访问（不会降低arithmetic intensity）
  - 不会损失data reuse
  - 不会需要更多sync
  - 没有大幅增加与结果无关的计算，如地址计算和流水线rotation

归根结底，TLP不是能不能增加的问题，而是是否要用目前单warp的执行效率换更多warp的trade-off。不过切换warp是没有overhead的，无需考虑这个。

### ILP

ILP可以说是在TLP的框架下存在的，不同的Occupancy可能需要不同的ILP优化方法，因为需要掩盖的延迟有时也会被TLP掩盖，比如double buffer有时会失效。ILP也相对来说更少trade-off，比如SASS-level schedule指令几乎没有成本，比如SMEM multi-stage和register prefetch一般都是用空闲资源。有几种情况可以很大程度倾向于ILP：

- 已经有很多独立的指令了，比如SGEMM里的FFMA
- latency较短或fixed
- 更多occupancy会降低reuse，因为这常常会伴随着
  - 更低的arithmetic intensity
  - 更多SMEM traffic
  - 更多sync，地址计算等等
- 提高occupancy的成本很高，但是这一点在实现做实验之前也很难说清楚。

### In our case

到底是使用1 CTA还是2 CTAs或许需要实验，来最终确定放弃一个CTA能否通过引入reuse增加concurrency。但简便起见，先与ref保持一致使用一个CTA。

修改计划是

- v4a增加register reuse，更大的thread tile
- v4b增加SMEM buffer
- v4c|d|e|...修改SASS-level scheduling达最优状态（如register prefetch和指令重排）

## v4 profiling的分析

原计划优化LDG+STS到掩盖大部分延迟，即v4全部用来处理warp stall，`v5`引入`cp.async`实验如下

- `v4a`在TLP和ILP之间选择ILP，放弃2 CTAs路线，选择单一CTA，增加thread tile至8*16。warp stall大幅降低，尤其是`Not Selected`、`Short Scoreboard`、`Barrier`、`MIO Throttle`。有了更多reuse和更多独立指令，编译器可以自动掩盖更多延迟。更高的Arithmetic Intensity使得内存相关的stall大幅减少。
  - 对于`Barrier`，其一：SM上总warp数少了，更少的warp在做STS操作，barrier周围也就疏通了一些。其二：`Barrier`的计算`Stall Barrier = (cycles a warp sits at barriers) / (instructions that warp issues)`是所有指令的平均，而增加thread tile后，每个k tile的barrier数不变，而指令大幅增加，分摊了stall。值得一提的是，barrier需要不同CTA掩盖，但原本2 CTA也不能很好掩盖，所以这也是一个走向1 CTA+ILP的原因。

- `v4b`增加了SMEM buffer，降低了一点`Barrier`，但schedule失误，仍使用了两个`__syncthreads()`
- `v4c`增加了register prefetch，并修改schedule为一个`__syncthreads()`，降低了`Barrier`、`Short Scoreboard`

现在`Dispatch Stall`、`LG Throttle`，`Short Scoreboard`成为了最显著的Stall，`Dispatch Stall`和FFMA pipe有关且CUTLASS ref不比我们的低，一般不会去优化`Dispatch Stall`，所以目标应该是`LG Throttle`和`Short Scoreboard`、`MIO Throttle`。

### LG Throttle

与global memory相关，在SASS中究其原因，是STG的问题，需要修改epilogue。优化到现在，可以看到以前不是瓶颈的问题成为了瓶颈。

`axpby(alpha, tCrC, beta, tCgC)`对应STG指令，每个thread tile都有128条STG。虽然在TiledMMA中已经用`PermutationMNK`使每4个元素相邻，但编译器仍使用了scalar store指令。到NCU source中查看，发现是从这lower到STG和FMUL的：

```cpp
CUTE_UNROLL
for (int i = 0; i < size(x); ++i) {
  if (p(i)) {
    y(i) = (isBetaZero ? alpha * x(i) : alpha * x(i) + beta * y(i));
  }
}
```

x对应tCrC，y对应tCgC，`axpby`默认lower到了单个元素的加法或乘法，在目前beta为0时，得到的SASS就是STG和FMUL的组合，没有LDG。故在后续修改中，无需考虑让epilogue经过SMEM，简便一些。

### Short Scoreboard

`Short Scoreboard`仍暴露了一部分。46.29%在一条BRA指令，剩余的~50%分布在FFMA指令和一条IMAD指令，其中IMAD指令是与之前的STS指令WAR产生的stall，不是等待LDS结果。

高`Short Scoreboard`的BRA指令是跳转至下一tile，也就是while大循环，stall全部来源于之前的指令结束，大部分LDS和一条BAR指令，对应`__syncthreads()`。来自于LDS的input scoreboard dependency有18条，包括预取下一个register block的指令。这里有一点需要纠正，scoreboard记录的不一定是寄存器数据依赖，更为准确的mental model是：

> scoreboard记录某个操作的完成点，有时与数据依赖有关，有时只为了保证指令的顺序性。它是编译器的规划本，一切都在编译期间生成，所以这些dependency都是static的。long或short scoreboard只和调用的硬件pipeline有关。

编译器把所有指令都schedule在一个tile结束，即BRA指令，预取下一个tile的block也会在这里结束，原因来自于inner loop中register prefetch的设计。在现在的设计中，计算与load交替，在一个tile内FFMA可以几乎完美掩盖LDS的latency：

```
LDS next_block;
FFMA current_block;
```

编译器可以在一个tile内尽可能提前LDS，用大量FFMA掩盖延迟，scoreboard dependency可以有很大的距离。但到了tile最后，也就是回到下一个tile开头的backedge前：

```
LDS next_tile.block0
FFMA current_tile.block7
BRA outer_loop
LDS next_tile.block1
FFMA next_tile.block0
```

新的一组LDS马上出现在了BRA之后，于是编译器保守地在BRA之前把scoreboard清空了，防止前一tile的LDS影响后续执行，在tile变多时对整体执行造成累计的影响。

另外，Input Scoreboard Dependencies每条都有2K条Attributed Stalls，而总数却少于$19 \times 2000 = 38000$，是因为他们共用了一个scoreboard slot。

### MIO Throttle

STS后紧接着LDS，重新schedule较复杂，考虑引入`cp.async`

## v4的FIX

### LG Throttle -> v4d

GMEM中C使用了`auto dC = make_stride(N, _1{});`，即N-major，而A和B自始至终都是M-major的，TiledMMA也为了配合s2r copy使用M-major，所以`make_tiled_copy_C`后retile和partition会生成第一个mode不contiguous的layout：

```
tXrD: ((_4,_4),_2,_4):((_16,_1),_64,_4)
tXgD: ((_4,_4),_2,_4):((_256,_1),_16384,_64)
```

无法自动向量化拷贝。

设计R2G TiledCopy实现transpose的copy，因为寄存器的访问特性，transpose实际上几乎没有成本。想像整个128\*128 tile的transpose，每个4\*4的小块翻转，每个线程对应的小块不变，因为数据本身排列是正确的，修改的只是TV排列。所以结果如下

```cpp
auto copy_epi =
  make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, TC>{},
                  Layout<Shape<_16, _16>, Stride<_1, _16>>{},
                  Layout<Shape<_4, _4>, Stride<_4, _1>>{});
```

- Thread Layout以整个256线程块作为最小单位，排列不变
- Value Layout翻转每个4\*4小块的TV排列

形象地来说，我们依然在处理TiledMMA规划的64\*64数据，但为了把数据向量化搬回GMEM，我们以n-major的视角看待每个线程处理的最小单位4\*4。

另外，该TiledCopy甚至可以服务G2R的copy，即原C mat的值搬运回register做运算，这实现了完整的epilogue copy。

### Short Scoreboard -> v4e

为了防止在BRA后的第一组LDS对scoreboard产生压力，尝试在S2R copy前后分别执行一部分计算，也就是：

```cpp
FFMA prefix of current_tile.block7
LDS next_tile.block0
FFMA suffix of current_tile.block7
BRA
FFMA prefix of next_tile.block0
LDS next_tile.block1
FFMA suffix of next_tile.block0
```

这样两组LDS之间的FFMA数量理论上不变，而BRA之后仍有一部分缓冲，prefix和suffix的比例可能需要仔细尝试，如果prefix过多，掩盖不了LDS的latency，如果suffix过多，不能很好的缓冲BRA。

首先尝试了GPT提出的“prefix用A/B的第一个值与B/A的所有值做乘法”，本意是用最少的计算告诉编译器：该block的所有值都在这里有用，就可以让LDS跨过BRA。结果显示确实BRA的input scoreboard dependency大幅减少，部分转移给FFMA，而跨tile的register prefetch转移到了iteration中第一个LDS上，但BRA和先前的LDS之间FFMA过少，short scoreboard stall依然暴露。从scoreboard看，编译器想要在第一组LDS复用第一个slot，所以依然是想要清除当前的scoreboard。

目前的schedule解决了分支的问题，这是一大进步。下一步尝试用更精细的边界schedule来解决stall，根据目前的现状，自然会想到增加tile边界的FFMA，对边界特殊处理（`k_block == K_BLOCK_MAX - 1`和`k_block == 0`），考虑在block 1~7放弃计算split的schedule以降低复杂度：

```
Blocks 1–6:
    LDS next block
    all FFMA current block

Block 7:
    rotate SMEM stage
    LDS next_tile.block0
    all FFMA current_tile.block7
    BRA
    
Block 0:
    large FFMA prefix of block0
    pipeline-maintenance work
    LDS block1
    remaining FFMA suffix of block0
```

short scoreboard stall仍然不下降，原因没有想清楚，但是发现寄存器使用已经达到了255，或许需要先引入`cp.async`减轻压力。scheduling优化先搁置。



### MIO Throttle -> v5

该修改基于v4d。引入了`cp.async`，却发现LDGSTS取代了STS的位置，MIO Throttle依然存在，这是schedule的问题。



## v5a profiling的分析（one final experiment）

目前的问题是epilogue占用了过多的寄存器，可能影响v4e整体schedule的效率，是否要继续优化v4？暂定不需要，因为引入`cp.async`是必然的，引入后大概率要重新调整schedule。

v5a去掉了epilogue里多余的register。

截至最新的v5a，性能达到了cublas的97%以上。剩余的3%性能与SM86和SIMT高度相关，以后再来探索吧~

处理好实验的结束是关键：

- 最后一次优化前的分析
- 优化时的假设
- 解释实验结果：是否有实现假设，产生了什么新的问题，性能差距为何存在

### 最后一次优化前的分析

#### MIO Throttle

集中在一组连续的`LDS`和`LDGSTS`，在inner loop中，register prefetch和`k_block == 0`的异步拷贝太近

#### Short Scoreboard

FFMA或LEA上的72.95%：最高stall的两个FFMA距离LDS 60行左右，且它们的一个operand对应的LDS受高MIO Throttle影响，short scoreboard是MIO Throttle造成的附带结果；很多FFMA的一个operand在对应的LDS后仅40行左右，由于低Occupancy不能完美覆盖。

BRA上的27.05%：input有5条LDS，是下一tile第一个register block prefetch的指令，与compiler保证指令顺序相关。Short Scoreboard的Not Issue Samples中BRA占比也不低，最后LDS和BRA之间有56条指令，应该足够完成拷贝，只能推测有几条LDS指令延迟较长。

### 优化时的假设

使用更深的register prefetch，让LDS和对应FFMA之间的依赖距离变长，给编译器更大空间重排指令，这样LDS和LDGSTS指令会分开，LDS latency应该更好被掩盖。

### 解释实验结果

TFLOPS有所上升，从cublas 92%上升到97%。Compute Throughput成功达到80%。整体指标都有优化，warp stall下降4%左右。

|                  | v5a  | v5a (1)         |
| ---------------- | ---- | --------------- |
| Not Selected     | 0.86 | 0.76 (-11.64%)  |
| Short Scoreboard | 0.09 | 0.22 (+152.52%) |
| Barrier          | 0.09 | 0.09 (+2.43%)   |
| MIO Throttle     | 0.09 | 0.04 (-58.14%)  |
| Wait             | 0.06 | 0.01 (-88.19%)  |

MIO Throttle和Wait由于更大指令重排空间降低，Short Scoreboard却与预期相反大幅上升。SASS中，编译器照常interleave了LDS和FFMA，可以看到相较于原来LDS更分散，但是一部分LDS之间的窗口更窄了。再看回目前的指标，可以说明编译器做了取舍，使用更多寄存器提升了性能，让更多latency暴露，LDS latency取代MIO burst成为新的瓶颈。

Short Scoreboard出现在FFMA和LEA（取代BRA成为一个tile的sync点），下一步可能是细致的重排`k_block == 0`时的schedule，使第一组（load block1）load有更多FFMA缓冲。

现在仍把schedule交给了编译器，下一步将会是有自己掌控。
