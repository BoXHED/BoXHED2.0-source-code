file(GLOB_RECURSE CPU_SOURCES *.cc *.h)
list(REMOVE_ITEM CPU_SOURCES ${boxhed_kernel_SOURCE_DIR}/src/cli_main.cc)

#-- Object library
# Object library is necessary for jvm-package, which creates its own shared library.
add_library(objboxhed_kernel OBJECT)
target_sources(objboxhed_kernel PRIVATE ${CPU_SOURCES})
# Skip files with factory object
set_source_files_properties(
  predictor/predictor.cc gbm/gbm.cc tree/tree_updater.cc metric/metric.cc objective/objective.cc
  PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
target_sources(objboxhed_kernel PRIVATE ${RABIT_SOURCES})

if (USE_CUDA)
  file(GLOB_RECURSE CUDA_SOURCES *.cu *.cuh)
  target_sources(objboxhed_kernel PRIVATE ${CUDA_SOURCES})
  boxhed_kernel_set_cuda_flags(objboxhed_kernel)
  target_link_libraries(objboxhed_kernel PRIVATE GPUTreeShap)
endif (USE_CUDA)

target_include_directories(objboxhed_kernel
  PRIVATE
  ${boxhed_kernel_SOURCE_DIR}/include
  ${boxhed_kernel_SOURCE_DIR}/dmlc-core/include
  ${boxhed_kernel_SOURCE_DIR}/rabit/include)
target_compile_options(objboxhed_kernel
  PRIVATE
  $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<COMPILE_LANGUAGE:CXX>>:/MP>
  $<$<AND:$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<COMPILE_LANGUAGE:CXX>>:-funroll-loops>)
if (WIN32 AND MINGW)
  target_compile_options(objboxhed_kernel PUBLIC -static-libstdc++)
endif (WIN32 AND MINGW)

if (MSVC)
  target_compile_options(objboxhed_kernel PRIVATE
    $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/utf-8>
    -D_CRT_SECURE_NO_WARNINGS
    -D_CRT_SECURE_NO_DEPRECATE
  )
endif (MSVC)

if (ENABLE_ALL_WARNINGS)
  target_compile_options(objboxhed_kernel PUBLIC
    $<IF:$<COMPILE_LANGUAGE:CUDA>,-Xcompiler=-Wall -Xcompiler=-Wextra,-Wall -Wextra>)
endif (ENABLE_ALL_WARNINGS)
if (LOG_CAPI_INVOCATION)
  target_compile_definitions(objboxhed_kernel PUBLIC -DLOG_CAPI_INVOCATION=1)
endif (LOG_CAPI_INVOCATION)

set_target_properties(objboxhed_kernel PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED ON)
target_compile_definitions(objboxhed_kernel
  PRIVATE
  -DDMLC_LOG_CUSTOMIZE=1  # enable custom logging
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:_MWAITXINTRIN_H_INCLUDED>)
if (USE_DEBUG_OUTPUT)
  target_compile_definitions(objboxhed_kernel PRIVATE -DXGBOOST_USE_DEBUG_OUTPUT=1)
endif (USE_DEBUG_OUTPUT)

if (XGBOOST_MM_PREFETCH_PRESENT)
  target_compile_definitions(objboxhed_kernel
    PRIVATE
    -DXGBOOST_MM_PREFETCH_PRESENT=1)
endif(XGBOOST_MM_PREFETCH_PRESENT)
if (XGBOOST_BUILTIN_PREFETCH_PRESENT)
  target_compile_definitions(objboxhed_kernel
    PRIVATE
    -DXGBOOST_BUILTIN_PREFETCH_PRESENT=1)
endif (XGBOOST_BUILTIN_PREFETCH_PRESENT)

find_package(Threads REQUIRED)
target_link_libraries(objboxhed_kernel PUBLIC Threads::Threads ${CMAKE_THREAD_LIBS_INIT})

if (USE_OPENMP OR USE_CUDA)  # CUDA requires OpenMP
  find_package(OpenMP REQUIRED)
  target_link_libraries(objboxhed_kernel PUBLIC OpenMP::OpenMP_CXX)
endif (USE_OPENMP OR USE_CUDA)

# For MSVC: Call msvc_use_static_runtime() once again to completely
# replace /MD with /MT. See https://github.com/dmlc/boxhed_kernel/issues/4462
# for issues caused by mixing of /MD and /MT flags
msvc_use_static_runtime()

# This grouping organises source files nicely in visual studio
auto_source_group("${CUDA_SOURCES}")
auto_source_group("${CPU_SOURCES}")

#-- End object library
