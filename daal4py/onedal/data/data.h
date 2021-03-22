/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#pragma once

#ifdef _WIN32
    #define NOMINMAX
#endif

#include <iostream>
#include <stdexcept>
#include <vector>
#include <limits>
#include <string>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "oneapi/dal/table/common.hpp"

namespace oneapi::dal::python
{
#if PY_VERSION_HEX < 0x03000000
    #define PyUnicode_Check(_x)      PyString_Check(_x)
    #define PyUnicode_AsUTF8(_x)     PyString_AsString(_x)
    #define PyUnicode_FromString(_x) PyString_FromString(_x)
#endif

#define SET_NPY_FEATURE(_T, _M, _E)                                                                                      \
    switch (_T)                                                                                                          \
    {                                                                                                                    \
    case NPY_DOUBLE:                                                                                                     \
    case NPY_CDOUBLE:                                                                                                    \
    case NPY_DOUBLELTR:                                                                                                  \
    case NPY_CDOUBLELTR:                                                                                                 \
    {                                                                                                                    \
        _M(double);                                                                                                      \
        break;                                                                                                           \
    }                                                                                                                    \
    case NPY_FLOAT:                                                                                                      \
    case NPY_CFLOAT:                                                                                                     \
    case NPY_FLOATLTR:                                                                                                   \
    case NPY_CFLOATLTR:                                                                                                  \
    {                                                                                                                    \
        _M(float);                                                                                                       \
        break;                                                                                                           \
    }                                                                                                                    \
    case NPY_INT:                                                                                                        \
    case NPY_INTLTR:                                                                                                     \
    {                                                                                                                    \
        _M(int);                                                                                                         \
        break;                                                                                                           \
    }                                                                                                                    \
    case NPY_UINT:                                                                                                       \
    case NPY_UINTLTR:                                                                                                    \
    {                                                                                                                    \
        _M(unsigned int);                                                                                                \
        break;                                                                                                           \
    }                                                                                                                    \
    case NPY_LONG:                                                                                                       \
    case NPY_LONGLTR:                                                                                                    \
    {                                                                                                                    \
        _M(long);                                                                                                        \
        break;                                                                                                           \
    }                                                                                                                    \
    case NPY_ULONG:                                                                                                      \
    case NPY_ULONGLTR:                                                                                                   \
    {                                                                                                                    \
        _M(unsigned long);                                                                                               \
        break;                                                                                                           \
    }                                                                                                                    \
    default: throw std::invalid_argument(std::string("Unsupported NPY type ") + std::to_string(_T) + " ignored\n."); _E; \
    };

oneapi::dal::table _input_to_onedal_table(PyObject * nda);

PyObject * _table_to_numpy(const oneapi::dal::table & input);

} // namespace oneapi::dal::python

#endif // _ONEDAL4PY_DATA_H_
