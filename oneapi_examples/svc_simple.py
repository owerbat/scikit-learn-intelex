from daal4py.onedal.svm import SVC
import numpy as np
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]], dtype=np.float32)
X2 = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]], dtype=np.float32)
y = np.array([1, 1, 2, 2], dtype=np.float32)

print(SVC)

from pprint import pprint
pprint(vars(SVC))


print(SVC(gamma='auto'))
clf = SVC(gamma='auto').fit(X, y)

print('dasd')

print(X)

print("PREDICT ", clf.predict(X))
print("PREDICT ", clf.decision_function(X))
