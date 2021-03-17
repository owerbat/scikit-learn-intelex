

# cov and SVC in new API
import daal4py as d4p
from daal4py.onedal.svm import SVC
from daal4py.onedal.primitive import cov
import numpy as np
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]], dtype=float)
y = np.array([1, 1, 2, 2], dtype=float)


print(cov(X))

clf = SVC(gamma='auto').fit(X, y)
print(clf.decision_function(X))

# KMeans in old API

init_alg = d4p.kmeans_init(nClusters = 10, fptype = "float", method = "randomDense")

centroids = init_alg.compute(data).centroids

alg = d4p.kmeans(nClusters = 10, maxIterations = 50, fptype = "float",
                 accuracyThreshold = 0, assignFlag = False)

result = alg.compute(data, centroids)
print(result)



from dpctx import device_context, device_type, memory

with device_context(device_type.gpu, 0):
    # copy data to gpu
    ms = memory.MemoryUSMShared(X.shape[0] * X.shape[1])
    ms.copy_from_host(X)
    X_gpu = np.ndarray(X.shape, buffer=ms, dtype=float)
    ms = memory.MemoryUSMShared(y.shape[0] * y.shape[1])
    ms.copy_from_host(y)
    y_gpu = np.ndarray(y.shape, buffer=ms, dtype=float)

    # daal4py algorithm
    kern = d4p.kernel_function_linear()
    train_algo = d4p.svm_training(method='thunder', kernel=kern, cacheSize=600000000)
    train_result = train_algo.compute(X_gpu, y_gpu)
    predict_algo = d4p.svm_prediction(kernel=kern, resultToEvulate='decision_function')
    print(predict_algo.compute(pdata, train_result.model))

