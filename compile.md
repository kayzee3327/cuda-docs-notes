# Cublas

在编译时，仅仅在CMakeLists.txt加上`target_link_libraries(${TARGET_NAME} PRIVATE CUDA::cublas)`，不一定能使进程找到cublas，如果路径没有硬编码在可执行文件，则需要查找LD_LIBRARY_PATH来定位`libcublas.so`等，此处提供两种硬编码方法

```cmake
# Embed the cuBLAS runtime path so the binary finds it without LD_LIBRARY_PATH
set_target_properties(${TARGET_NAME} PROPERTIES
    BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}"
    INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}"
)

set_target_properties(${TARGET_NAME} PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH TRUE
)
```

RPATH - Runtime Search Path 

- 第一种方法使用了`find_package(CUDAToolkit REQUIRED)`的结果，两者搭配使用。当build或`make install`时将对应路径硬编码到可执行文件中
- 第二种方法
  - `INSTALL_RPATH_USE_LINK_PATH`：告诉cmake将每个shared library对应的路径硬编码在可执行文件里
  - `BUILD_WITH_INSTALL_RPATH`：通常cmake只会在install时硬编码RPATH，这一指令告诉cmake在build时也硬编码

## 若不成功

即使指挥CMake编译RPATH，仍然有可能在编译时没有硬编码。

那么此时，强行查找

```cmake
target_link_libraries(${TARGET_NAME} PRIVATE CUDA::cublas)
# Derive the real library directory from the cublas import target and embed it
# as an rpath so the binary runs without LD_LIBRARY_PATH.
get_target_property(_cublas_loc CUDA::cublas IMPORTED_LOCATION)
get_filename_component(_cublas_dir "${_cublas_loc}" DIRECTORY)
set_target_properties(${TARGET_NAME} PROPERTIES
    BUILD_RPATH  "${_cublas_dir}"
    INSTALL_RPATH "${_cublas_dir}"
    )
```



# Profiling with Nsight

大多数情况下，使用`-lineinfo`而不是`-G`。CUTLASS/CuTe使用了大量模板，如果`-G`禁用了所有优化，对GPU CPU以及两者之间的数据传输压力很大。尤其是外接显示器时，当GPU既要渲染桌面又被ncu指挥着一遍遍重新运行kernel，Windows Desktop Window Manager (DWM) 发现一段时间没有渲染桌面，它就会尝试重置gpu驱动，现象体现为一段时间黑屏。此时极易引起WSL和Windows的死锁。
