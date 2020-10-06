"""
The 'daal4py.sklearn.neighbors' module implements the k-nearest neighbors
algorithm.
"""


from ._classification import KNeighborsClassifier
from ._regression import KNeighborsRegressor
from ._unsupervised import NearestNeighbors

__all__ = ['KNeighborsClassifier', 'KNeighborsRegressor', 'NearestNeighbors']
