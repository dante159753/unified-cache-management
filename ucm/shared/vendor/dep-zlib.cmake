set(SKIP_INSTALL_ALL ON CACHE INTERNAL "" FORCE)
set(ZLIB_BUILD_TESTS OFF CACHE INTERNAL "" FORCE)
set(ZLIB_BUILD_EXAMPLES OFF CACHE INTERNAL "" FORCE)

FetchContent_Declare(zlib
    URL ${UCM_VENDOR_ARCHIVE_DIR}/zlib-1.3.1.tar.gz
    URL_HASH SHA256=9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23
)
FetchContent_MakeAvailable(zlib)
target_compile_options(zlibstatic PRIVATE -fPIC)
