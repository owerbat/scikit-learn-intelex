
from daal4py import SVC
import numpy as np
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]], dtype=float)
y = np.array([1, 1, 2, 2], dtype=float)

print(SVC)

from pprint import pprint
pprint(vars(SVC))


print(SVC(gamma='auto'))
print(SVC(gamma='auto').fit(X, y))
