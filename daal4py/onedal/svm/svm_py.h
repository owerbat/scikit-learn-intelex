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

#include "oneapi/dal/algo/svm.hpp"
#include "data/data.h"

namespace oneapi::dal::python
{
struct classification_svm_params
{
    std::string kernel;
    std::string method;
    double c;
    double accuracy_threshold;
    int max_iteration_count;
    double cache_size;
    bool shrinking;
    double shift;
    double scale;
    int degree;
    double sigma;
};

// clas <task>_<algorithm>
class classification_svm_train
{
public:
    // from descriptor
    classification_svm_train(classification_svm_params * params);

    // attributes from train_input
    void train(PyObject * data, PyObject * labels, PyObject * weights);

    // attributes from train_result
    int get_support_vector_count();

    // attributes from train_result
    PyObject * get_support_vectors();

    // attributes from train_result
    PyObject * get_support_indices();

    // attributes from train_result
    PyObject * get_coeffs();

    // attributes from train_result
    PyObject * get_biases();

private:
    classification_svm_params params_;
    svm::train_result<svm::task::classification> train_result_;
};

// // clas <task>_<algorithm>
class classification_svm_infer
{
public:
    // from descriptor
    classification_svm_infer(classification_svm_params * params);

    // attributes from infer_input.hpp expect model
    void infer(PyObject * data, PyObject * support_vectors, PyObject * coeffs, PyObject * biases);

    // attributes from infer_result
    PyObject * get_labels();

    // attributes from infer_result
    PyObject * get_decision_function();

private:
    classification_svm_params params_;
    svm::infer_result<svm::task::classification> infer_result_;
};

} // namespace oneapi::dal::python
