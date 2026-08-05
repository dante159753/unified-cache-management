find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE TORCH_QUERY_RESULT
)
if(NOT TORCH_QUERY_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to locate torch CMake files")
endif()
list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH})

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYBIND11_QUERY_RESULT
)
if(NOT PYBIND11_QUERY_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to locate pybind11 CMake files")
endif()
list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_CMAKE_PATH})

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pathlib, torch_npu; print(pathlib.Path(torch_npu.__file__).parent)"
    OUTPUT_VARIABLE TORCH_NPU_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE TORCH_NPU_QUERY_RESULT
)
if(NOT TORCH_NPU_QUERY_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to locate torch_npu")
endif()

find_package(Torch REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

if(NOT ASCEND_HOME_PATH)
    if(DEFINED ENV{ASCEND_HOME_PATH})
        set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
    elseif(ASCEND_ROOT)
        set(ASCEND_HOME_PATH ${ASCEND_ROOT})
    else()
        set(ASCEND_HOME_PATH /usr/local/Ascend/ascend-toolkit/latest)
    endif()
endif()

if(EXISTS ${ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake was not found under ${ASCEND_HOME_PATH}")
endif()

set(ASCEND_CANN_PACKAGE_PATH ${ASCEND_HOME_PATH})
if(NOT SOC_VERSION)
    set(SOC_VERSION ascend910b3)
endif()
include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

ascendc_library(ucm_segmented_copy_kernels SHARED
    csrc/op_kernel/segmented_copy.cpp
)
ascendc_library(ucm_compact_scatter_kernels SHARED
    csrc/op_kernel/compact_scatter.cpp
)

pybind11_add_module(ucm_segmented_copy
    csrc/op_host/torch_binding.cpp
)
target_include_directories(ucm_segmented_copy PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_PATH}/include
    ${ASCEND_HOME_PATH}/include
)
target_link_directories(ucm_segmented_copy PRIVATE
    ${TORCH_NPU_PATH}/lib
    ${ASCEND_HOME_PATH}/lib64
)
target_link_libraries(ucm_segmented_copy PRIVATE
    ucm_segmented_copy_kernels
    ucm_compact_scatter_kernels
    ${TORCH_LIBRARIES}
    torch_npu
    ascendcl
)
target_compile_options(
    ucm_segmented_copy PRIVATE ${TORCH_CXX_FLAGS} -Wno-unused-function
)
set_target_properties(ucm_segmented_copy PROPERTIES
    BUILD_RPATH "$ORIGIN;${TORCH_NPU_PATH}/lib;${ASCEND_HOME_PATH}/lib64"
)

file(RELATIVE_PATH INSTALL_REL_PATH ${UCM_ROOT_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
install(
    TARGETS ucm_segmented_copy ucm_segmented_copy_kernels ucm_compact_scatter_kernels
    LIBRARY DESTINATION ${INSTALL_REL_PATH}
    COMPONENT ucm
)
