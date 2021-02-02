#define NO_IMPORT_ARRAY
#include <cstdint>
#include <cstring>
#include <Python.h>
#include "data.h"

#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

#include "daal.h"

using namespace oneapi;

#define is_array(a)           ((a) && PyArray_Check(a))
#define array_type(a)         PyArray_TYPE((PyArrayObject *)a)
#define array_is_behaved(a)   (PyArray_ISCARRAY_RO((PyArrayObject *)a) && array_type(a) < NPY_OBJECT)
#define array_is_behaved_F(a) (PyArray_ISFARRAY_RO((PyArrayObject *)a) && array_type(a) < NPY_OBJECT)
#define array_is_native(a)    (PyArray_ISNOTSWAPPED((PyArrayObject *)a))
#define array_numdims(a)      PyArray_NDIM((PyArrayObject *)a)
#define array_data(a)         PyArray_DATA((PyArrayObject *)a)
#define array_size(a, i)      PyArray_DIM((PyArrayObject *)a, i)

class NumpyDeleter
{
public:
    NumpyDeleter(PyArrayObject * a) : _ndarray(a) {}

    NumpyDeleter(const NumpyDeleter & o) : _ndarray(o._ndarray) {}

    void operator()(const void * ptr)
    {
        // We need to protect calls to python API
        // Note: at termination time, even when no threads are running, this breaks without the protection
        PyGILState_STATE gstate = PyGILState_Ensure();
        assert((void *)array_data(_ndarray) == ptr);
        Py_DECREF(_ndarray);
        PyGILState_Release(gstate);
    }
    // We don't want this to be copied
    NumpyDeleter & operator=(const NumpyDeleter &) = delete;

private:
    PyArrayObject * _ndarray;
};

template <typename T>
inline dal::homogen_table _make_ht(PyObject * nda)
{
    PyArrayObject * array = reinterpret_cast<PyArrayObject *>(nda);
    assert(is_array(nda) && array_is_behaved(array));

    if (array_numdims(array) == 2)
    {
        printf("[CPP]: _make_ht: array_numdims(array) == 2\n");
        T * data_pointer          = reinterpret_cast<T *>(array_data(array));
        const size_t row_count    = static_cast<size_t>(array_size(array, 0));
        const size_t column_count = static_cast<size_t>(array_size(array, 1));
        auto res_table            = dal::homogen_table(data_pointer, row_count, column_count, NumpyDeleter(array));
        // we need it increment the ref-count if we use the input array in-place
        // if we copied/converted it we already own our own reference
        if (reinterpret_cast<PyObject *>(data_pointer) == nda) Py_INCREF(array);
        printf("[CPP]: _make_ht: finish\n");

        return res_table;
    }
    else if (array_numdims(array) == 1)
    {
        printf("[CPP]: _make_ht: array_numdims(array) == 1\n");
        T * data_pointer          = reinterpret_cast<T *>(array_data(array));
        const size_t row_count    = static_cast<size_t>(array_size(array, 0));
        const size_t column_count = 1;
        auto res_table            = dal::homogen_table(data_pointer, row_count, column_count, NumpyDeleter(array));
        // we need it increment the ref-count if we use the input array in-place
        // if we copied/converted it we already own our own reference
        if (reinterpret_cast<PyObject *>(data_pointer) == nda) Py_INCREF(array);
        printf("[CPP]: _make_ht: finish\n");

        return res_table;
    }
    else
    {
        throw std::runtime_error("Input array has wrong dimensionality (must be 2d).");
    }
}

dal::table _input_to_onedal_table(PyObject * obj)
{
    dal::table res;
    if (PyErr_Occurred())
    {
        PyErr_Print();
        PyErr_Clear();
    }
    if (obj == nullptr || obj == Py_None)
    {
        throw std::invalid_argument("obj is nullptr");
    }
    if (is_array(obj))
    { // we got a numpy array
        PyArrayObject * ary = (PyArrayObject *)obj;
        printf("[CPP]: _input_to_onedal_table: is_array\n");
        if (array_is_behaved(ary))
        {
            switch (PyArray_DESCR(ary)->type)
            {
            case NPY_DOUBLE:
            case NPY_CDOUBLE:
            case NPY_DOUBLELTR:
            case NPY_CDOUBLELTR: res = _make_ht<double>(obj); break;
            case NPY_FLOAT:
            case NPY_CFLOAT:
            case NPY_FLOATLTR:
            case NPY_CFLOATLTR: res = _make_ht<float>(obj); break;
            default: throw std::invalid_argument("Not avalible type in input_to_onedal_table.");
            }

            // #define MAKENT_(_T) res = _make_ht<_T>(obj)
            //             SET_NPY_FEATURE(PyArray_DESCR(ary)->type, MAKENT_, throw std::invalid_argument("Found unsupported array type"));
            // #undef MAKENT_
            printf("[CPP]: _input_to_onedal_table: finish\n");
        }
        // throw std::invalid_argument("Could not convert Python object to onedal table.");
    }
    return res;
}

class VSP
{
public:
    // we need a virtual destructor
    virtual ~VSP() {};
};
// typed virtual shared pointer, for simplicity we make it a oneDAL shared pointer
template <typename T>
class TVSP : public VSP, public daal::services::SharedPtr<T>
{
public:
    TVSP(const daal::services::SharedPtr<T> & org) : daal::services::SharedPtr<T>(org) {}
    virtual ~TVSP() {};
};

void daalsp_free_cap(PyObject * cap)
{
    VSP * sp = static_cast<VSP *>(PyCapsule_GetPointer(cap, NULL));
    if (sp)
    {
        delete sp;
        sp = NULL;
    }
}

template <typename T>
void set_sp_base(PyArrayObject * ary, daal::services::SharedPtr<T> & sp)
{
    void * tmp_sp  = static_cast<void *>(new TVSP<T>(sp));
    PyObject * cap = PyCapsule_New(tmp_sp, NULL, daalsp_free_cap);
    PyArray_SetBaseObject(ary, cap);
}

// Uses a shared pointer to a raw array (T*) for creating a nd-array
template <typename T, int NPTYPE>
static PyObject * _sp_to_nda(daal::services::SharedPtr<T> & sp, std::int64_t nr, std::int64_t nc)
{
    npy_intp dims[2] = { static_cast<npy_intp>(nr), static_cast<npy_intp>(nc) };
    PyObject * obj   = PyArray_SimpleNewFromData(2, dims, NPTYPE, static_cast<void *>(sp.get()));
    if (!obj) throw std::invalid_argument("conversion to numpy array failed");
    set_sp_base(reinterpret_cast<PyArrayObject *>(obj), sp);
    printf("[_sp_to_nda] finish\n");
    return obj;
}

template <typename T>
struct daal_object_owner
{
    explicit daal_object_owner(const T & obj) : obj_(obj) {}

    void operator()(const void *) { obj_ = T {}; }

    T obj_;
};

PyObject * _table_to_numpy(const dal::table & input)
{
    if (!input.has_data())
    {
        throw std::invalid_argument("Empty data");
    }
    if (input.get_kind() == dal::homogen_table::kind())
    {
        const auto & homogen_res = static_cast<const dal::homogen_table &>(input);
        if (homogen_res.get_data_layout() == dal::data_layout::row_major)
        {
            const auto & dtype = homogen_res.get_metadata().get_data_type(0);

            switch (dtype)
            {
            case dal::data_type::float32:
            {
                auto rows = dal::row_accessor<const float> { homogen_res }.pull();
                rows.need_mutable_data();
                auto daal_data = daal::services::SharedPtr<float>(rows.get_mutable_data(), daal_object_owner { rows });
                printf("[_table_to_numpy float32]: _sp_to_nda\n");
                return _sp_to_nda<float, NPY_FLOAT32>(daal_data, homogen_res.get_row_count(), homogen_res.get_column_count());
            }
            case dal::data_type::float64:
            {
                auto rows = dal::row_accessor<const double> { homogen_res }.pull();
                rows.need_mutable_data();
                auto daal_data = daal::services::SharedPtr<double>(rows.get_mutable_data(), daal_object_owner { rows });
                printf("[_table_to_numpy float64]: _sp_to_nda\n");
                return _sp_to_nda<double, NPY_FLOAT64>(daal_data, homogen_res.get_row_count(), homogen_res.get_column_count());
            }

                // TODO: other types
                // TODO. How own data without copy?
            }
        }
    }
    return nullptr;
}
