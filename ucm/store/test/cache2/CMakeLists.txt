# CtrlLayout is still a stub on feature_a5. Compile unchanged DataStrategy sources
# next to a test control block, without changing the production class/interface.
set(CACHE2_SOURCE_DIR ${UCM_ROOT_DIR}/ucm/store/cache2/cc)
set(CACHE2_FIXTURE_DIR ${CMAKE_CURRENT_BINARY_DIR}/fixture)
configure_file(${CACHE2_SOURCE_DIR}/data_strategy.h ${CACHE2_FIXTURE_DIR}/data_strategy.h COPYONLY)
configure_file(${CACHE2_SOURCE_DIR}/data_strategy.cc ${CACHE2_FIXTURE_DIR}/data_strategy.cc COPYONLY)
configure_file(mock/ctrl_layout.h ${CACHE2_FIXTURE_DIR}/ctrl_layout.h COPYONLY)

# Keep the fake SDK and Device::Setup isolated from the production trans target.
add_executable(cache2_hal.test
    hal_data_strategy_test.cc
    ${CACHE2_FIXTURE_DIR}/data_strategy.cc
    ${UCM_ROOT_DIR}/ucm/shared/trans/ascend/hal/hal_host_buffers.cc
)
target_compile_definitions(cache2_hal.test PRIVATE UCM_RUNTIME_ASCEND_HAL=1)
target_include_directories(cache2_hal.test PRIVATE
    ${CACHE2_FIXTURE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/mock
    ${UCM_ROOT_DIR}/ucm/shared
)
target_link_libraries(cache2_hal.test PRIVATE infra_status infra_logger gtest_main gmock)
gtest_discover_tests(cache2_hal.test NO_PRETTY_VALUES PROPERTIES LABELS cache2_hal)
