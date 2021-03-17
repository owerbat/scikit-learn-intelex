#ifndef _SVM_PY_INCLUDED_
#define _SVM_PY_INCLUDED_

#include "oneapi/dal/algo/svm.hpp"
#include "data_management/data.h"

namespace oneapi::dal::py_svm
{
// clas <task>_<algorithm>
class classification_svm
{
public:
    // from descriptor
    classification_svm(std::string, double, double, int, double, bool);

    // attributes from train_input
    void train_cpp(PyObject * data, PyObject * labels, PyObject * weights);

    // attributes from train_result
    int get_support_vector_count();

    // attributes from train_result
    PyObject * get_support_vectors();

    // attributes from train_result
    PyObject * get_support_indices();

    // attributes from train_result
    PyObject * get_coeffs();

    // attributes from train_result
    double get_bias();

    // attributes from infer_input.hpp expect model
    void infer_cpp(PyObject * data);

    // attributes from infer_result
    PyObject * get_labels();

    // attributes from infer_result
    PyObject * get_decision_function();

private:
    svm::train_result<svm::task::classification> train_result_;
    svm::infer_result<svm::task::classification> infer_result_;
    dal::svm::descriptor<double, dal::svm::method::thunder, dal::svm::task::classification> descriptor_; // TODO
};

} // namespace oneapi::dal::py_svm

#endif // _SVM_PY_INCLUDED_
