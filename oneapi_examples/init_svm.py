
from daal4py.onedal.svm import SVC
import numpy as np
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]], dtype=float)
X2 = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]], dtype=float)
y = np.array([1, 1, 2, 2], dtype=float)

print(SVC)

from pprint import pprint
pprint(vars(SVC))


print(SVC(gamma='auto'))


# GPU
ctx1:
    clf = SVC(gamma='auto').fit(X, y)

    # 1 case!
    # HOST
    # Copy memory
    print("PREDICT ", clf.predict(X2))


# 2 case

with ctx_gpu:
    clf = SVC(gamma='auto').fit(X, y)


a = b * 2

with ctx_gpu:
    print("PREDICT ", clf.predict(X2))

# if can we use same memory on GPU

# !!! 3 case (will work automatic)

with ctx_gpu:
    clf = SVC(gamma='auto').fit(X, y)


a = b * 2

print("PREDICT ", clf.predict(X2))


# !!!! 4 case (not working)
# user side? we can make it, but can we errors/segfaults.

# copy daata.

clf = SVC(gamma='auto').fit(X, y)


a = b * 2

with ctx_gpu:
    print("PREDICT ", clf.predict(X2))


# 5 case

# copy daata. bad case. DPCRT has default context?

with ctx_cpu:
    clf = SVC(gamma='auto').fit(X, y)
a = b * 2

with ctx_gpu:
    print("PREDICT ", clf.predict(X2))


# story info about context where whey was allocate
# we can use this info


# 2 case
# CPU / HOST
ctx2:
    print("PREDICT ", clf.decision_function(X2))

print('FIT END')

# print('PREDICT END')
# print("decision_function ", clf.decision_function(X2))



# from daal4py.onedal.svm import SVC
# from daal4py.onedal.svm import SVC
# from onedal4py.svm import SVC


# data managments:

