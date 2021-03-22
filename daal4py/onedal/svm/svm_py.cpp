#define NO_IMPORT_ARRAY

#ifdef _DPCPP_
    #include "dpctl_sycl_types.h"
    #include "dpctl_sycl_queue_manager.h"
#endif

#include "svm_py.h"
#include "common/utils.h"

namespace oneapi::dal::python
{
// template <typename fptype>
// inline auto & get_descriptor_fptype(classification_svm_params & params)
// {
//     if (params.kernel == "linear")
//     {
//         auto kernel_desc = linear_kernel::descriptor<fptype> {};
//         if (params.method == "smo")
//         {
//             return svm::descriptor<fptype, svm::method::smo, svm::task::classification, linear_kernel::descriptor<fptype> > { kernel_desc };
//         }
//         else if (params.method == "thunder")
//         {
//             return svm::descriptor<fptype, svm::method::thunder, svm::task::classification, linear_kernel::descriptor<fptype> > { kernel_desc };
//         }
//     }
//     else if (params.kernel == "rbf")
//     {
//         auto kernel_desc = rbf_kernel::descriptor<fptype> {};
//         if (params.method == "smo")
//         {
//             return svm::descriptor<fptype, svm::method::smo, svm::task::classification, rbf_kernel::descriptor<fptype> > { kernel_desc };
//         }
//         else if (params.method == "thunder")
//         {
//             return svm::descriptor<fptype, svm::method::thunder, svm::task::classification, rbf_kernel::descriptor<fptype> > { kernel_desc };
//         }
//     }
//     throw std::runtime_error("get_descriptor_fptype");
// }

inline const auto get_descriptor(classification_svm_params & params, data_type data_type_input)
{
    // if (data_type_input == data_type::float64)
    // {
    //     return get_descriptor_fptype<double>(params);
    // }
    // else if (data_type_input == data_type::float32)
    // {
    //     return get_descriptor_fptype<float>(params);
    // }
    auto kernel_desc = linear_kernel::descriptor<float> {};
    return svm::descriptor<float, svm::method::smo, svm::task::classification, linear_kernel::descriptor<float> > { kernel_desc };
}

// from descriptor
classification_svm_train::classification_svm_train(classification_svm_params * params) : params_(*params) {}

// attributes from train_input
void classification_svm_train::train(PyObject * data, PyObject * labels, PyObject * weights)
{
    thread_allow _allow;
    auto data_table    = _input_to_onedal_table(data);
    auto labels_table  = _input_to_onedal_table(labels);
    auto weights_table = _input_to_onedal_table(weights);
    auto data_type     = data_table.get_metadata().get_data_type(0);

    auto descriptor = get_descriptor(params_, data_type);
    train_result_   = python::train(descriptor, data_table, labels_table, weights_table);
    printf("[classification_svm] finish train\n");
}

// attributes from train_result
int classification_svm_train::get_support_vector_count()
{
    return train_result_.get_support_vector_count();
}

// attributes from train_result
PyObject * classification_svm_train::get_support_vectors()
{
    return _table_to_numpy(train_result_.get_support_vectors());
}

// attributes from train_result
PyObject * classification_svm_train::get_support_indices()
{
    return _table_to_numpy(train_result_.get_support_indices());
}

// attributes from train_result
PyObject * classification_svm_train::get_coeffs()
{
    return _table_to_numpy(train_result_.get_coeffs());
}

// attributes from train_result
PyObject * classification_svm_train::get_biases()
{
    return _table_to_numpy(train_result_.get_biases());
}

// from descriptor
classification_svm_infer::classification_svm_infer(classification_svm_params * params) : params_(*params) {}

// attributes from infer_input.hpp expect model
void classification_svm_infer::infer(PyObject * data, PyObject * support_vectors, PyObject * coeffs, PyObject * biases)
{
    thread_allow _allow;
    printf("classification_svm::infer\n");

    auto data_table            = _input_to_onedal_table(data);
    auto support_vectors_table = _input_to_onedal_table(support_vectors);
    auto coeffs_table          = _input_to_onedal_table(coeffs);
    auto biases_table          = _input_to_onedal_table(biases);

    auto data_type  = data_table.get_metadata().get_data_type(0);
    auto descriptor = get_descriptor(params_, data_type);

    auto model = svm::model<svm::task::classification> {}
                     .set_support_vectors(support_vectors_table)
                     .set_coeffs(coeffs_table)
                     .set_biases(biases_table)
                     .set_first_class_label(0)
                     .set_second_class_label(1);
    infer_result_ = python::infer(descriptor, model, data_table);
}

// attributes from infer_result
PyObject * classification_svm_infer::get_labels()
{
    return _table_to_numpy(infer_result_.get_labels());
}

// attributes from infer_result
PyObject * classification_svm_infer::get_decision_function()
{
    return _table_to_numpy(infer_result_.get_decision_function());
}

} // namespace oneapi::dal::python
