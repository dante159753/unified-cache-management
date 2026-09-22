if(NOT BUILD_UNIT_TESTS OR NOT UCM_RUNTIME_ASCEND_HAL)
    return()
endif()

find_package(Threads REQUIRED)

# CtrlLayout's inline placeholders cannot be intercepted at link time. Compile
# unchanged DataStrategy sources beside the test's control-block implementation.
set(DATA_STRATEGY_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../cache2/cc")
set(DATA_STRATEGY_FIXTURE_DIR "${CMAKE_CURRENT_BINARY_DIR}/fixture")
file(MAKE_DIRECTORY "${DATA_STRATEGY_FIXTURE_DIR}")
foreach(source data_strategy.cc data_strategy.h)
    configure_file("${DATA_STRATEGY_SOURCE_DIR}/${source}"
                   "${DATA_STRATEGY_FIXTURE_DIR}/${source}" COPYONLY)
endforeach()
configure_file(ctrl_layout.h "${DATA_STRATEGY_FIXTURE_DIR}/ctrl_layout.h" COPYONLY)

add_executable(cache2_data_strategy_a5.test
    data_strategy_a5_test.cc
    "${DATA_STRATEGY_FIXTURE_DIR}/data_strategy.cc"
)
target_include_directories(cache2_data_strategy_a5.test PRIVATE "${DATA_STRATEGY_FIXTURE_DIR}")
target_link_libraries(cache2_data_strategy_a5.test PRIVATE trans Threads::Threads)

add_test(NAME cache2.data_strategy_a5 COMMAND cache2_data_strategy_a5.test)
set_tests_properties(cache2.data_strategy_a5 PROPERTIES
    LABELS "ascend-a5;hardware"
    RUN_SERIAL TRUE
    PROCESSORS 2
    TIMEOUT 720
)
