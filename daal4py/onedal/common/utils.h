/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef _ONEDAL4PY_COMMON_UTILS_H_
#define _ONEDAL4PY_COMMON_UTILS_H_

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

namespace oneapi::dal::python
{
class thread_allow
{
public:
    thread_allow() { allow(); }
    ~thread_allow() { disallow(); }
    void allow() { _save = PyEval_SaveThread(); }
    void disallow()
    {
        if (_save)
        {
            PyEval_RestoreThread(_save);
            _save = NULL;
        }
    }

private:
    PyThreadState * _save;
};

template <typename... Args>
auto train(Args &&... args)
{
#ifdef _DPCPP_
    auto dpctl_queue = DPCTLQueueMgr_GetCurrentQueue();
    if (dpctl_queue != NULL)
    {
        cl::sycl::queue & sycl_queue = *reinterpret_cast<cl::sycl::queue *>(dpctl_queue);
        return dal::train(sycl_queue, std::forward<Args>(args)...);
    }
    else
    {
        throw std::runtime_error("Cannot set daal context: Pointer to queue object is NULL");
    }
#else
    return dal::train(std::forward<Args>(args)...);
#endif
}

template <typename... Args>
auto infer(Args &&... args)
{
#ifdef _DPCPP_
    auto dpctl_queue = DPCTLQueueMgr_GetCurrentQueue();
    if (dpctl_queue != NULL)
    {
        cl::sycl::queue & sycl_queue = *reinterpret_cast<cl::sycl::queue *>(dpctl_queue);
        return dal::infer(sycl_queue, std::forward<Args>(args)...);
    }
    else
    {
        throw std::runtime_error("Cannot set daal context: Pointer to queue object is NULL");
    }
#else
    return dal::infer(std::forward<Args>(args)...);
#endif
}

} // namespace oneapi::dal::python

#endif // _ONEDAL4PY_COMMON_UTILS_H_
