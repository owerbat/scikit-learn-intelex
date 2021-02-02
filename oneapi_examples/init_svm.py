
from daal4py import SVC
import numpy as np
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]], dtype=float)
X2 = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]], dtype=float)
y = np.array([1, 1, 2, 2], dtype=float)

print(SVC)

from pprint import pprint
pprint(vars(SVC))


print(SVC(gamma='auto'))
clf = SVC(gamma='auto').fit(X, y)

print('FIT END')

print("PREDICT ", clf.predict(X2))

print('PREDICT END')
