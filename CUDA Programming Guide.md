A good technical note should be a "trigger" for a concept, not a transcript of the manual.

# 1. Introduction to CUDA

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

**GPU memory**

Each device has its own memory

- CPU and GPU use a single unified virtual memory space
  - distinct address (between CPU and GPU)

- global/device memory: GPU DRAM, accessible to all SMs
- unified data cache: shared memory
  - finite size **per thread block** allocation
  - accessible to threads in a thread block or a thread block cluster
- unified data cache: L1 cache
- register file
  - finite size **per thread** allocation
- constant cache
  - stores constant value over a lifetime of a kernel

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

