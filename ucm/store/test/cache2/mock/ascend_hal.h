/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * */
#pragma once

#include <cstddef>
#include <cstdint>

// Test-only subset of the driver interface; no CANN installation is needed.
enum drvError_t { DRV_ERROR_NONE = 0, DRV_ERROR_INVALID_VALUE = 3 };
enum drv_mem_side { MEM_HOST_SIDE = 0, MEM_DEV_SIDE = 1, MEM_HOST_NUMA_SIDE = 2 };
enum drv_mem_pg_type { MEM_NORMAL_PAGE_TYPE = 0, MEM_HUGE_PAGE_TYPE = 1 };
enum drv_mem_type { MEM_HBM_TYPE = 0, MEM_DDR_TYPE = 1 };
enum drv_mem_handle_type { MEM_HANDLE_TYPE_NONE = 0 };
enum drv_mem_granularity_options {
    MEM_ALLOC_GRANULARITY_MINIMUM = 0,
    MEM_ALLOC_GRANULARITY_RECOMMENDED = 1
};
enum ShareHandleAttrType { SHR_HANDLE_ATTR_NO_WLIST_IN_SERVER = 0 };
inline constexpr unsigned int SHR_HANDLE_NO_WLIST_ENABLE = 1;

struct drv_mem_prop {
    uint32_t side;
    uint32_t devid;
    uint32_t module_id;
    uint32_t pg_type;
    uint32_t mem_type;
    uint64_t reserve;
};

struct ShareHandleAttr {
    unsigned int enableFlag;
    unsigned int rsv[8];
};

using drv_mem_handle_t = struct drv_mem_handle;

extern "C" {
drvError_t halMemGetAllocationGranularity(const drv_mem_prop* prop,
                                          drv_mem_granularity_options option, size_t* granularity);
drvError_t halMemAddressReserve(void** ptr, size_t size, size_t alignment, void* addr,
                                uint64_t flags);
drvError_t halMemAddressFree(void* ptr);
drvError_t halMemCreate(drv_mem_handle_t** handle, size_t size, const drv_mem_prop* prop,
                        uint64_t flags);
drvError_t halMemRelease(drv_mem_handle_t* handle);
drvError_t halMemMap(void* ptr, size_t size, size_t offset, drv_mem_handle_t* handle,
                     uint64_t flags);
drvError_t halMemUnmap(void* ptr);
drvError_t halMemExportToShareableHandle(drv_mem_handle_t* handle, drv_mem_handle_type type,
                                         uint64_t flags, uint64_t* shareHandle);
drvError_t halMemShareHandleSetAttribute(uint64_t shareHandle, ShareHandleAttrType type,
                                         ShareHandleAttr attr);
drvError_t halMemImportFromShareableHandle(uint64_t shareHandle, uint32_t deviceId,
                                           drv_mem_handle_t** handle);
}
