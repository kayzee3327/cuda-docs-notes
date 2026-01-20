A good technical note should be a "trigger" for a concept, not a transcript of the manual.

## 1.2 Programming model

### Describe components of  CUDA programming model  (in a programmer's view)

![](.\images\gpu-cpu-system-diagram.png)

Above diagram clearly depicts the conceptual model of a hardware supporting CUDA.

### A. The CUDA programming model assumes a heterogeneous computing system

**CPU, host**

- Host memory: system DRAM
- Components outside Computing Area
  - mem control
  - public L3 cache
  - PCIe or NVLINK
- Components inside Computing Area
  - CPU cores
    - a local register file
    - next-to-execution-area L2 cache
    - integrated L1 cache

**GPU, device**

- Device memory: GPU DRAM
- Components outside Computing Area
  - mem control
  - public L2 cache
  - PCIe or NVLINK
- Components inside Computing Area
  - Streaming Multiprocessors (SMs), organized into groups called **Graphics Processing Clusters** (GPCs)
    - a local register file
    - unified cache data, providing the physical resources for shared memory and L1 cache (allocation can be configured at runtime)
    - functional units that perform computations
      - CUDA cores, Tensor cores, Tensor memory, etc.

**GPU memory (containing section 2.2.3)**

CPU and GPU use a single unified virtual memory space. (Distinct address between CPU and GPU)

- global/device memory: GPU DRAM, accessible to all SMs
  - persistent until allocation is **freed** or throughout the lifetime of the application
  - possible data races between threads
  - from/to host
- unified data cache: shared memory
  - persistent throughout kernel execution
  - possible data races between threads
  - finite size **per thread block** allocation
  - accessible to threads in a thread block or a thread block cluster
- unified data cache: L1 cache
- register file
  - thread local scope
  - finite size **per thread** allocation
  - managed by compiler
  - not "addressable"
- local memory
  - thread local scope
  - managed by compiler
  - use global memory space
  - organized and prefer coalesced access
    - consecutive 32-bit words accessed by consecutive thread IDs
  - **allocated** when
    - variable-length arrays
    - large structures
    - register spilling (too much variables)
- constant memory
  - grid scope, persistent throughout the lifetime of the application (CUDA context)
  - read-only to kernels
  - must be declared globally with `__constant__`
  - use global memory space (typically 64KB)
  - has a distinct object per device 
  - has a dedicate cache
- Texture and Surface memory (uncommon)
- Distributed Shared Memory (>= sm90)
  - read, write and perform atomics across a cluster
  - only for communication
    - per thread block configuration (size, dynamic/static)
  - size = thread blocks $\times$ shared memory per block
  - access when all thread blocks exists (before any thread block exits)
    - ensure this with `cluster.sync()` (like `__syncthreads()`)

**when executing a CUDA application on such systems**

- CUDA application

  - host code, using CUDA APIs to copy data between the host memory and device memory
  - device code (a function that is invoked for execution on the GPU is called a **kernel**)

- Procedure

  1. **always** start execution on the CPU
  2. launch the kernel (starting many threads executing the kernel code in parallel on the GPU)

  The CPU and GPU can both be executing code **simultaneously**, and best performance is usually found by maximizing utilization of both CPUs and GPUs.

### B. CUDA Parallel Execution

**Launching**

- A grid (of thread blocks)
  - 1, 2, or 3 dimensional
  - could be arbitrarily large
- thread blocks (of threads)
  - 1, 2, or 3 dimensional
  - max size 1024 threads
  - each executes in a single SM, meaning threads in a block should be executed in the same SM
    - allow threads to communicate and synchronize efficiently
      - through on-chip shared memory, etc.
    - finish on that SM in most cases 
      - exceptions such as CUDA Dynamic Parallelism
  - must not rely on results from each other, meaning threads cannot rely on threads in different blocks
    - no guarantee of scheduling between blocks

Each thread can therefore locate itself and determine responsible data and operations.

- optional parameters
  - thread block cluster (>= sm90)
    - 1, 2, or 3 dimensional
    - no effect on thread block positioning in a grid, and have a position within the containing cluster
    - each executes in a single Graphics Processing Cluster (GPC)
      - thread blocks scheduled **simultaneously**
    - allow thread blocks to communicate and synchronize using software interfaces provided by **Cooperative Groups**
      - access to shared memory of all blocks in a cluster, naming **Distributed Shared Memory**
    - adjacent thread blocks
  - stream
  - SM configuration settings

**Execution on hardware**

- SIMT
- Threads in a thread block are organized into **warps**
  - 1 warp == 32 threads
  - warp lane (0~31)
  - each thread in a warp may follow different execution path (branches)
    - sometimes called **warp divergence**
    - GPU is max-utilized if no warp divergence
  - all threads in a warp execute same instruction **simultaneously**
    - branching: mask off inactive threads (condition false)
    - Hardware could optimize masked lanes. (Undefined behaviors if programming model violated)

- Thread blocks are best to have size multiple of 32.



## 1.3 The CUDA platform

### Main Component of CUDA applications and libraries

![](.\images\fatbin.png)

**cubin (CUDA binary)**

- for specific SM version

- compatibility

  - same major version
  - greater than or equal minor version

  Example: sm86 cubin can be loaded and executed on sm89 but not sm90

**fatbin (container of GPU code, ptx and cubin)**

- PTX 
  - for specific compute capability
  - can be JIT compiled for higher or equal compute capability
    - utilize new compiler optimization
    - enable forward compatibility without rebuilding

## 2.1 Intro to CUDA C++

### How is CUDA runtime initialized?

这里需要日后总结，实践经验缺乏

## 2.2 Writing CUDA SIMT Kernels

### How to maximize global memory performance?

**Access Pattern: aligned 32-byte transaction (256 bits)**

- When a thread need 4 bytes, the warp actually requests a aligned 32-byte transaction

**Coalesced Access**

- If multiple threads **in a warp** use data in a single transaction, the warp would coalesces memory accesses, resulting in less requests.
- Most straight forward: Consecutive threads access Consecutive data
  - less common
- More General: higher $\frac{\text{bytes used}}{\text{bytes transferred}}$
  -  For example, memory access is permuted but covers all 32 bytes

**辅助实验**

- 证明任取一个4字节，都会取出32字节
- 证明在warp内的乱序访问只要用上了全部32字节，仍等效于连续线程访问连续内存
- 证明跨warp的访问不会被合并



### How to maximize shared memory performance?

**Access Pattern**

- 32 shared memory banks
  - A bank: 32-bit bandwidth per clock cycle

**Avoiding bank conflicts**

- bank conflicts
  - different threads **in a warp** access different data in a same bank, resulting in serialization of access 
- read broadcasting: multiple threads **in a warp** access the word at one SMEM address
- "random" write: when multiple threads write to a same word, only one of threads **in a warp** write to the address

## 2.5 NVCC: The NVIDIA CUDA Compiler

**source file types**

- Host-only: `.c`, `.cpp`, `.cc`, `.cxx`
- Mixed (Need special configuration in editors): `.h`, `.hpp`, `.hh`, `.hxx`
- Mixed: `.cu`, `.cuh`

### Describe the NVCC Compilation Workflow

