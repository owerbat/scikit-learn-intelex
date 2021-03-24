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

#include "svm/svm_py.h"
#include "common/utils.h"
#include "common/train.h"
#include "common/infer.h"

namespace oneapi::dal::python
{
template <typename KernelDescriptor>
KernelDescriptor get_kernel_params(const classification_svm_params & params)
{
    if constexpr (std::is_same_v<typename KernelDescriptor::tag_t, rbf_kernel::detail::descriptor_tag>)
    {
        return KernelDescriptor {}.set_sigma(params.sigma);
    }
    return KernelDescriptor {};
}

template <typename Descriptor, typename... Args>
auto train_descriptor_impl(Descriptor descriptor, const classification_svm_params & params, Args &&... args)
{
    descriptor.set_c(params.c)
        .set_class_count(params.class_count)
        .set_accuracy_threshold(params.accuracy_threshold)
        .set_max_iteration_count(params.max_iteration_count)
        .set_cache_size(params.shrinking)
        .set_kernel(get_kernel_params<typename Descriptor::kernel_t>(params));

    return python::train(descriptor, std::forward<Args>(args)...);
}

template <typename... Args>
svm::train_result<svm::task::classification> train_impl(classification_svm_params & params, data_type data_type_input, Args &&... args)
{
    if (data_type_input == data_type::float32 && params.method == "smo" && params.kernel == "linear")
    {
        return train_descriptor_impl(svm::descriptor<float, svm::method::smo, svm::task::classification, linear_kernel::descriptor<float> > {},
                                     params, std::forward<Args>(args)...);
    }
    else if (data_type_input == data_type::float32 && params.method == "smo" && params.kernel == "rbf")
    {
        return train_descriptor_impl(svm::descriptor<float, svm::method::smo, svm::task::classification, rbf_kernel::descriptor<float> > {}, params,
                                     std::forward<Args>(args)...);
    }
    if (data_type_input == data_type::float32 && params.method == "smo" && params.kernel == "poly")
    {
        return train_descriptor_impl(svm::descriptor<float, svm::method::smo, svm::task::classification, polynomial_kernel::descriptor<float> > {},
                                     params, std::forward<Args>(args)...);
    }
    else if (data_type_input == data_type::float32 && params.method == "thunder" && params.kernel == "linear")
    {
        return train_descriptor_impl(svm::descriptor<float, svm::method::thunder, svm::task::classification, linear_kernel::descriptor<float> > {},
                                     params, std::forward<Args>(args)...);
    }
    else if (data_type_input == data_type::float32 && params.method == "thunder" && params.kernel == "rbf")
    {
        return train_descriptor_impl(svm::descriptor<float, svm::method::thunder, svm::task::classification, rbf_kernel::descriptor<float> > {},
                                     params, std::forward<Args>(args)...);
    }
    else if (data_type_input == data_type::float64 && params.method == "thunder" && params.kernel == "poly")
    {
        return train_descriptor_impl(
            svm::descriptor<float, svm::method::thunder, svm::task::classification, polynomial_kernel::descriptor<float> > {}, params,
            std::forward<Args>(args)...);
    }
    if (data_type_input == data_type::float64 && params.method == "smo" && params.kernel == "linear")
    {
        return train_descriptor_impl(svm::descriptor<double, svm::method::smo, svm::task::classification, linear_kernel::descriptor<double> > {},
                                     params, std::forward<Args>(args)...);
    }
    else if (data_type_input == data_type::float64 && params.method == "smo" && params.kernel == "rbf")
    {
        return train_descriptor_impl(svm::descriptor<double, svm::method::smo, svm::task::classification, rbf_kernel::descriptor<double> > {}, params,
                                     std::forward<Args>(args)...);
    }
    else if (data_type_input == data_type::float64 && params.method == "smo" && params.kernel == "poly")
    {
        return train_descriptor_impl(svm::descriptor<double, svm::method::smo, svm::task::classification, polynomial_kernel::descriptor<double> > {},
                                     params, std::forward<Args>(args)...);
    }
    else if (data_type_input == data_type::float64 && params.method == "thunder" && params.kernel == "linear")
    {
        return train_descriptor_impl(svm::descriptor<double, svm::method::thunder, svm::task::classification, linear_kernel::descriptor<double> > {},
                                     params, std::forward<Args>(args)...);
    }
    else if (data_type_input == data_type::float64 && params.method == "thunder" && params.kernel == "rbf")
    {
        return train_descriptor_impl(svm::descriptor<double, svm::method::thunder, svm::task::classification, rbf_kernel::descriptor<double> > {},
                                     params, std::forward<Args>(args)...);
    }
    else if (data_type_input == data_type::float64 && params.method == "thunder" && params.kernel == "poly")
    {
        return train_descriptor_impl(
            svm::descriptor<double, svm::method::thunder, svm::task::classification, polynomial_kernel::descriptor<double> > {}, params,
            std::forward<Args>(args)...);
    }

    else
    {
        return train_descriptor_impl(svm::descriptor<double, svm::method::smo, svm::task::classification, rbf_kernel::descriptor<double> > {}, params,
                                     std::forward<Args>(args)...);
    }
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

    train_result_ = train_impl(params_, data_type, data_table, labels_table, weights_table);
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

    auto data_type = data_table.get_metadata().get_data_type(0);
    // auto descriptor = get_descriptor(params_, data_type);

    auto model = svm::model<svm::task::classification> {}
                     .set_support_vectors(support_vectors_table)
                     .set_coeffs(coeffs_table)
                     .set_biases(biases_table)
                     .set_first_class_label(0)
                     .set_second_class_label(1);
    // infer_result_ = python::infer(descriptor, model, data_table);
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
