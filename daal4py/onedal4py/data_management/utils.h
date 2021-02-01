#ifndef _ONEDAL4PY_UTILS_H_
#define _ONEDAL4PY_UTILS_H_

#ifdef _WIN32
    #define NOMINMAX
#endif
#include <string>
#include <numpy/arrayobject.h>

static std::string to_std_string(PyObject * o)
{
    return PyUnicode_AsUTF8(o);
}

#endif
