#define NO_IMPORT_ARRAY

#ifdef _DPCPP_
    #include "dpctl_sycl_types.h"
    #include "dpctl_sycl_queue_manager.h"
#endif

#include "svm_py.h"

namespace oneapi::dal::py_svm
{
// from descriptor
classification_svm::classification_svm(std::string kernel, double c, double accuracy_threshold, int max_iteration_count, double cache_size, bool shrinking)
{
    descriptor_.set_c(c).set_accuracy_threshold(accuracy_threshold).set_max_iteration_count(max_iteration_count).set_cache_size(cache_size);
}

// attributes from train_input
void classification_svm::train_cpp(PyObject * data, PyObject * labels, PyObject * weights)
{
    printf("[classification_svm] start train\n");

    auto data_table   = _input_to_onedal_table(data);
    auto labels_table = _input_to_onedal_table(labels);
// auto weights_table = _input_to_onedal_table(weights);
#ifdef _DPCPP_
    auto dpctl_queue = DPCTLQueueMgr_GetCurrentQueue();
    if (dpctl_queue != NULL)
    {
        cl::sycl::queue & sycl_queue = *reinterpret_cast<cl::sycl::queue *>(dpctl_queue);
        train_result_                = dal::train(sycl_queue, descriptor_, data_table, labels_table);
    }
    else
    {
        throw std::runtime_error("Cannot set daal context: Pointer to queue object is NULL");
    }
#else
    train_result_ = dal::train(descriptor_, data_table, labels_table);
#endif
    printf("[classification_svm] finish train\n");
}

// attributes from train_result
int classification_svm::get_support_vector_count()
{
    return train_result_.get_support_vector_count();
}

// attributes from train_result
PyObject * classification_svm::get_support_vectors()
{
    return _table_to_numpy(train_result_.get_support_vectors());
}

// attributes from train_result
PyObject * classification_svm::get_support_indices()
{
    return _table_to_numpy(train_result_.get_support_indices());
}

// attributes from train_result
PyObject * classification_svm::get_coeffs()
{
    return _table_to_numpy(train_result_.get_coeffs());
}

// attributes from train_result
double classification_svm::get_bias()
{
    return train_result_.get_bias();
}

// attributes from infer_input.hpp expect model
void classification_svm::infer_cpp(PyObject * data)
{
    auto data_table = _input_to_onedal_table(data);
    printf("classification_svm::infer\n");
#ifdef _DPCPP_
    auto dpctl_queue = DPCTLQueueMgr_GetCurrentQueue();
    if (dpctl_queue != NULL)
    {
        cl::sycl::queue & sycl_queue = *reinterpret_cast<cl::sycl::queue *>(dpctl_queue);
        infer_result_                = dal::infer(sycl_queue, descriptor_, train_result_.get_model(), data_table);
    }
    else
    {
        throw std::runtime_error("Cannot set daal context: Pointer to queue object is NULL");
    }
#else
    infer_result_ = dal::infer(descriptor_, train_result_.get_model(), data_table);
#endif
}

// attributes from infer_result
PyObject * classification_svm::get_labels()
{
    return _table_to_numpy(infer_result_.get_labels());
}

// attributes from infer_result
PyObject * classification_svm::get_decision_function()
{
    return _table_to_numpy(infer_result_.get_decision_function());
}

} // namespace oneapi::dal::py_svm
