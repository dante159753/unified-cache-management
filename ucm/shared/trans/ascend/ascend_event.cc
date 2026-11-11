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
#include <acl/acl.h>
#include <dlfcn.h>
#include "trans/event.h"

namespace UC::Trans {
namespace {

using ValueOp = aclError (*)(void*, uint64_t, uint32_t, aclrtStream);

struct ExternalEventApi {
    ValueOp wait{reinterpret_cast<ValueOp>(dlsym(RTLD_DEFAULT, "aclrtValueWait"))};
    ValueOp write{reinterpret_cast<ValueOp>(dlsym(RTLD_DEFAULT, "aclrtValueWrite"))};
};

const ExternalEventApi& GetExternalEventApi()
{
    static const ExternalEventApi api;
    return api;
}

struct ExternalEventWaitStream {
    aclrtStream stream{nullptr};
    ~ExternalEventWaitStream()
    {
        if (stream != nullptr) { aclrtDestroyStream(stream); }
    }
};

}  // namespace

bool Event::SupportsExternalWait() noexcept
{
    const auto& api = GetExternalEventApi();
    return api.wait != nullptr && api.write != nullptr;
}

Status Event::WaitExternal() const
{
    if (!Valid()) { return Status::InvalidParam("missing external event handle"); }
    if (!SupportsExternalWait()) { return Status::Unsupported(); }

    // A dump dispatcher stays on one device. Consume the event on one stream
    // before submitting copies, which may be spread across many streams.
    thread_local ExternalEventWaitStream waitStream;
    if (waitStream.stream == nullptr) {
        auto ret = aclrtCreateStream(&waitStream.stream);
        if (ret != ACL_SUCCESS) { return Status{ret, std::to_string(ret)}; }
    }
    const auto& api = GetExternalEventApi();
    auto* addr = reinterpret_cast<void*>(NativeHandle());
    constexpr uint32_t waitEqual = 1;
    auto ret = api.wait(addr, 1, waitEqual, waitStream.stream);
    if (ret != ACL_SUCCESS) { return Status{ret, std::to_string(ret)}; }
    const auto reset = api.write(addr, 0, 0, waitStream.stream);
    ret = aclrtSynchronizeStream(waitStream.stream);
    if (reset != ACL_SUCCESS) { return Status{reset, std::to_string(reset)}; }
    if (ret != ACL_SUCCESS) { return Status{ret, std::to_string(ret)}; }
    return Status::OK();
}

Status Event::Synchronize() const
{
    if (!Valid()) { return Status::OK(); }
    const auto ret = aclrtSynchronizeEvent(reinterpret_cast<aclrtEvent>(NativeHandle()));
    return ret == ACL_SUCCESS ? Status::OK() : Status{ret, std::to_string(ret)};
}

Expected<bool> Event::Query() const
{
    if (!Valid()) { return true; }
    aclrtEventRecordedStatus state = ACL_EVENT_RECORDED_STATUS_NOT_READY;
    const auto ret = aclrtQueryEventStatus(reinterpret_cast<aclrtEvent>(NativeHandle()), &state);
    if (ret != ACL_SUCCESS) { return Status{ret, std::to_string(ret)}; }
    return state == ACL_EVENT_RECORDED_STATUS_COMPLETE;
}

}  // namespace UC::Trans
