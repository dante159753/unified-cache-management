if(BUILD_NUMA)
    message(STATUS "Building numactl library...")

    set(NUMA_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/numa_install)
    FetchContent_Declare(
        numactl
        URL ${UCM_VENDOR_ARCHIVE_DIR}/numactl-2.0.16.tar.gz
        URL_HASH SHA256=1b242f893af977a1d31af6ce9d6b8dafdd2d8ec3dc9207f7c2dc0d3446e7c7c8
    )
    FetchContent_MakeAvailable(numactl)
    if(NOT EXISTS "${NUMA_INSTALL_DIR}/lib/libnuma.so")
        message(STATUS "Configuring numactl...")
        execute_process(
            COMMAND ./configure --prefix=${NUMA_INSTALL_DIR}
            WORKING_DIRECTORY ${numactl_SOURCE_DIR}
            RESULT_VARIABLE numa_configure_result
            OUTPUT_VARIABLE numa_configure_output
            ERROR_VARIABLE numa_configure_error
        )
        if(NOT numa_configure_result EQUAL 0)
            message(FATAL_ERROR "Failed to configure numactl. \n"
                                "Result: ${numa_configure_result}\n"
                                "STDOUT: ${numa_configure_output}\n"
                                "STDERR: ${numa_configure_error}\n")
        endif()

        message(STATUS "Building and installing numactl...")
        execute_process(
            COMMAND make install -j8
            WORKING_DIRECTORY ${numactl_SOURCE_DIR}
            RESULT_VARIABLE numa_install_result
            OUTPUT_VARIABLE numa_install_output
            ERROR_VARIABLE numa_install_error
        )
        if(NOT numa_install_result EQUAL 0)
            message(FATAL_ERROR "Failed to build and install numactl. \n"
                                "Result: ${numa_install_result}\n"
                                "STDOUT: ${numa_install_output}\n"
                                "STDERR: ${numa_install_error}\n")
        endif()
    else()
        message(STATUS "Found already built libnuma. Skipping build.")
    endif()

    add_definitions(-DNUMA_ENABLED)
else()
    message(STATUS "Skipping numactl build...")
endif()

string(TOLOWER "$ENV{PLATFORM}" PLATFORM_ENV)
if(RUNTIME_ENVIRONMENT STREQUAL "cuda")
    message(STATUS "Building GSAOnDevice for CUDA...")
    add_subdirectory(csrc/cuda/hash_retrieval)
    add_subdirectory(csrc/cuda/ham_dist)
endif()
