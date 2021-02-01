
#include "svm_py.h"
#include "oneapi/dal/algo/svm.hpp"
#include "data_management/data.h"

#include "dppl_sycl_types.h"
#include "dppl_sycl_queue_manager.h"

namespace oneapi::dal::py_svm
{
// clas <task>_<algorithm>
class classification_svm
{
public:
    // from descriptor
    classification_svm(std::string kernel, double c, double accuracy_threshold, int max_iteration_count, double cache_size, bool shrinking) {}

    // attributes from train_input
    void train(PyObject * data, PyObject * labels, PyObject * weights)
    {
        auto dppl_queue = DPPLQueueMgr_GetCurrentQueue();
        if (dppl_queue != NULL)
        {
            cl::sycl::queue * sycl_queue = reinterpret_cast<cl::sycl::queue *>(dppl_queue);
            auto data_table              = _input_to_onedal_table(queue, data);
            auto labels_table            = _input_to_onedal_table(queue, labels);
            auto weights_table           = _input_to_onedal_table(queue, weights);

            train_result_ = dal::train(queue, svm_desc, data_table, labels_table, weights_table);
        }
        else
        {
            throw std::runtime_error("Cannot set daal context: Pointer to queue object is NULL");
        }
    }

    // attributes from train_result
    int get_support_vector_count() { return train_result_->get_support_vector_count(); }

    // attributes from train_result
    PyObject * get_support_vectors() {}

    // attributes from train_result
    PyObject * get_support_indices() {}

    // attributes from train_result
    PyObject * get_coeffs() {}

    // attributes from train_result
    double get_bias() {}

    // attributes from infer_input.hpp expect model
    void infer(PyObject * data)
    {
        auto dppl_queue = DPPLQueueMgr_GetCurrentQueue();
        if (dppl_queue != NULL)
        {
            cl::sycl::queue * queue = reinterpret_cast<cl::sycl::queue *>(dppl_queue);

            auto data_table = _input_to_onedal_table(queue, data);
            infer_result_   = dal::train(queue, svm_desc, train_result_->get_model(), data_table);
        }
        else
        {
            throw std::runtime_error("Cannot set daal context: Pointer to queue object is NULL");
        }
    }

    // attributes from infer_result
    PyObject * get_labels() { return infer_result_.get_labels(); }

    // attributes from infer_result
    PyObject * get_decision_function() { return infer_result_.get_decision_function(); }

private:
    svm::train_result<svm::task::classification> train_result_;
    svm::infer_result<svm::task::classification> infer_result_;
};

} // namespace oneapi::dal::py_svm
