find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

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
find_package(pybind11 CONFIG REQUIRED)

set(UCM_ALLGATHER_INSTALL_TARGETS allgatherstore ucm_allgather_runtime)

if(UCM_RUNTIME_ASCEND_FAMILY)
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
        if(RUNTIME_ENVIRONMENT STREQUAL "ascend-a3")
            set(SOC_VERSION ascend910_9392)
        else()
            set(SOC_VERSION ascend910b3)
        endif()
    endif()
    message(STATUS "AllGather AscendC SOC_VERSION=${SOC_VERSION}")
    include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

    ascendc_library(ucm_segmented_copy_kernels SHARED
        csrc/op_kernel/segmented_copy.cpp
    )
    ascendc_library(ucm_compact_scatter_kernels SHARED
        csrc/op_kernel/compact_scatter.cpp
    )

    add_library(allgatherplatform STATIC cc/platform_ascend.cc)
    set_target_properties(allgatherplatform PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(allgatherplatform PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/cc
        ${ASCEND_HOME_PATH}/include
    )
    target_link_directories(allgatherplatform PUBLIC ${ASCEND_HOME_PATH}/lib64)
    target_link_libraries(allgatherplatform PUBLIC
        infra_status
        ucm_segmented_copy_kernels
        ucm_compact_scatter_kernels
        ascendcl
        hccl
    )

    list(APPEND UCM_ALLGATHER_INSTALL_TARGETS
        ucm_segmented_copy_kernels
        ucm_compact_scatter_kernels
    )
elseif(RUNTIME_ENVIRONMENT STREQUAL "cuda")
    add_library(allgatherplatform STATIC cc/platform_cuda.cu)
    set_target_properties(allgatherplatform PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_ARCHITECTURES "75;80;86;89;90"
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )
    target_include_directories(allgatherplatform PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/cc
    )
    target_link_libraries(allgatherplatform PUBLIC infra_status cudart)
else()
    message(FATAL_ERROR "AllGatherStore requires Ascend or CUDA runtime")
endif()

file(GLOB UCM_ALLGATHER_STORE_CC_SOURCE_FILES "./cc/*.cc")
list(FILTER UCM_ALLGATHER_STORE_CC_SOURCE_FILES EXCLUDE REGEX "platform_ascend\\.cc$")
add_library(allgatherstore SHARED ${UCM_ALLGATHER_STORE_CC_SOURCE_FILES})
target_include_directories(allgatherstore PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cc)
target_link_libraries(allgatherstore PRIVATE
    storeintf
    infra_logger
    metrics
    allgatherplatform
)

pybind11_add_module(ucm_allgather_runtime
    csrc/op_host/runtime_binding.cpp
    cc/memory_plan.cc
)
target_include_directories(ucm_allgather_runtime PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cc)
target_link_libraries(ucm_allgather_runtime PRIVATE allgatherplatform)

file(RELATIVE_PATH INSTALL_REL_PATH ${UCM_ROOT_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
install(
    TARGETS ${UCM_ALLGATHER_INSTALL_TARGETS}
    LIBRARY DESTINATION ${INSTALL_REL_PATH}
    COMPONENT ucm
)
