#===============================================================================
# Copyright 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from abc import ABCMeta, abstractmethod
import numbers
from math import ceil

import numpy as np
from scipy import sparse as sp
from ..datatypes import (
    _validate_targets,
    _check_X_y,
    _check_array,
    _check_is_fitted,
    _column_or_1d,
    _check_n_features
)

from ..common._policy import _HostPolicy
from ..datatypes._data_conversion import from_table, to_table
from onedal import _backend


class BaseForest(BaseEstimator, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self,
                 n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf,
                 min_weight_fraction_leaf, max_features, max_leaf_nodes,
                 min_impurity_decrease, min_impurity_split, bootstrap, oob_score,
                 random_state, warm_start, class_weight, ccp_alpha, max_samples,
                 max_bins, min_bin_size, infer_mode, voting_mode, error_metric_mode,
                 variable_importance_mode, algorithm):
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state
        self.warm_start = warm_start
        self.class_weight = class_weight
        self.max_samples = max_samples
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.ccp_alpha = ccp_alpha
        self.max_bins = max_bins
        self.min_bin_size = min_bin_size
        self.infer_mode = infer_mode
        self.voting_mode = voting_mode
        self.error_metric_mode = error_metric_mode
        self.variable_importance_mode = variable_importance_mode
        self.algorithm = algorithm

    def _to_absolute_max_features(max_features, n_features, is_classification=False):
        if max_features is None:
            return n_features
        elif isinstance(max_features, str):
            if max_features == "auto":
                return max(1, int(np.sqrt(n_features))
                           ) if is_classification else n_features
            elif max_features == 'sqrt':
                return max(1, int(np.sqrt(n_features)))
            elif max_features == "log2":
                return max(1, int(np.log2(n_features)))
            else:
                raise ValueError(
                    'Invalid value for max_features. Allowed string '
                    'values are "auto", "sqrt" or "log2".')
        elif isinstance(max_features, (numbers.Integral, np.integer)):
            return max_features
        else:
            if max_features > 0.0:
                return max(1, int(max_features * n_features))
            return 0

    def _get_observations_per_tree_fraction(n_samples, max_samples):
        if max_samples is None:
            return 1.

        if isinstance(max_samples, numbers.Integral):
            if not (1 <= max_samples <= n_samples):
                msg = "`max_samples` must be in range 1 to {} but got value {}"
                raise ValueError(msg.format(n_samples, max_samples))
            return float(max_samples / n_samples)

        if isinstance(max_samples, numbers.Real):
            if not (0 < float(max_samples) <= 1):
                msg = "`max_samples` must be in range (0.0, 1.0] but got value {}"
                raise ValueError(msg.format(max_samples))
            return float(max_samples)

        msg = "`max_samples` should be int or float, but got type '{}'"
        raise TypeError(msg.format(type(max_samples)))

    def _get_onedal_params(self, data):
        features_per_node = self._to_absolute_max_features(
            self.max_features, data.shape[1], self.is_classification)

        observations_per_tree_fraction = self._get_observations_per_tree_fraction(
            n_samples=data.shape[0], max_samples=self.max_samples)

        min_observations_in_leaf_node = (self.min_samples_leaf
                                         if isinstance(
                                             self.min_samples_leaf, numbers.Integral)
                                         else int(ceil(
                                             self.min_samples_leaf * data.shape[0])))

        min_observations_in_split_node = (self.min_samples_split
                                          if isinstance(
                                              self.min_samples_split, numbers.Integral)
                                          else int(ceil(
                                              self.min_samples_split * data.shape[0])))

        return {
            'fptype': 'float' if data.dtype is np.dtype('float32') else 'double',
            'method': self.algorithm,
            'class_count': 0 if self.classes_ is None else len(self.classes_),
            'infer_mode': self.infer_mode,
            'voting_mode': self.voting_mode,
            'observations_per_tree_fraction': observations_per_tree_fraction,
            'impurity_threshold': float(
                0.0 if self.min_impurity_split is None else self.min_impurity_split),
            'min_weight_fraction_in_leaf_node': self.min_weight_fraction_leaf,
            'min_impurity_decrease_in_split_node': self.min_impurity_decrease,
            'tree_count': int(self.n_estimators),
            'features_per_node': features_per_node,
            'max_tree_depth': int(0 if self.max_depth is None else self.max_depth),
            'min_observations_in_leaf_node': min_observations_in_leaf_node,
            'min_observations_in_split_node': min_observations_in_split_node,
            'max_leaf_nodes': (0 if self.max_leaf_nodes is None else self.max_leaf_nodes),
            'max_bins': self.max_bins,
            'min_bin_size': self.min_bin_size,
            'memory_saving_mode': False,
            'bootstrap': bool(self.bootstrap),
            'error_metric_mode': self.error_metric_mode,
            'variable_importance_mode': self.variable_importance_mode,
        }

    def _fit(self, X, y, sample_weight, module):
        pass

    def _predict(self, X, module):
        pass

    def _predict_proba(self, X, module):
        pass


class RandomForestClassifier(BaseForest):
    def __init__(self,
                 n_estimators=100,
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=True,
                 oob_score=False,
                 random_state=None,
                 warm_start=False,
                 class_weight=None,
                 ccp_alpha=0.0,
                 max_samples=None,
                 max_bins=256,
                 min_bin_size=1,
                 infer_mode='class_responses',
                 voting_mode='weighted',
                 error_metric_mode='none',
                 variable_importance_mode='none',
                 algorithm='hist'):
        super().__init__(
            n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
            max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split, bootstrap=bootstrap,
            oob_score=oob_score, random_state=random_state, warm_start=warm_start,
            class_weight=class_weight, ccp_alpha=ccp_alpha, max_samples=max_samples,
            max_bins=max_bins, min_bin_size=min_bin_size, infer_mode=infer_mode,
            voting_mode=voting_mode, error_metric_mode=error_metric_mode,
            variable_importance_mode=variable_importance_mode, algorithm=algorithm)
        self.is_classification = True

    def fit(self, X, y, sample_weight=None):
        return super()._fit(X, y, sample_weight, _backend.decision_forest.classification)

    def predict(self, X):
        pred = super()._predict(X, _backend.decision_forest.classification)
        return np.take(self.classes_, pred.ravel().astype(np.int64, casting='unsafe'))

    def predict_proba(self, X):
        pass


class RandomForestRegressor(BaseForest):
    def __init__(self,
                 n_estimators=100,
                 criterion="squared_error",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=True,
                 oob_score=False,
                 random_state=None,
                 warm_start=False,
                 class_weight=None,
                 ccp_alpha=0.0,
                 max_samples=None,
                 max_bins=256,
                 min_bin_size=1,
                 infer_mode='class_responses',
                 voting_mode='weighted',
                 error_metric_mode='none',
                 variable_importance_mode='none',
                 algorithm='hist'):
        super().__init__(
            n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
            max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split, bootstrap=bootstrap,
            oob_score=oob_score, random_state=random_state, warm_start=warm_start,
            class_weight=class_weight, ccp_alpha=ccp_alpha, max_samples=max_samples,
            max_bins=max_bins, min_bin_size=min_bin_size, infer_mode=infer_mode,
            voting_mode=voting_mode, error_metric_mode=error_metric_mode,
            variable_importance_mode=variable_importance_mode, algorithm=algorithm)
        self.is_classification = False

    def fit(self, X, y, sample_weight=None):
        return super()._fit(X, y, sample_weight, _backend.decision_forest.regression)

    def predict(self, X):
        return super()._predict(X, _backend.decision_forest.regression).ravel()
