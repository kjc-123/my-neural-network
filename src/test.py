# unit tests

import mynetwork
import loader

import unittest

import numpy as np

class TestNetworkMethods(unittest.TestCase):
    def test_forward_prop(self):
        W1 = np.random.randn(10, 9)
        I1 = np.random.randn(9, 1)
        b1 = np.random.randn(10, 1)
        result = (W1 @ I1) + b1
        self.assertEqual(mynetwork.MyNetwork.forwardprop(I1, W1, b1).all(), result.all())
    def test_cost(self):
        y = np.array([3, 4, 7])
        yhat = np.array([-2, -2, 8])
        result = 62
        self.assertEqual(mynetwork.MyNetwork.cost(yhat, y), result)
    def test_sigmoid(self):
        z = np.random.randn(16, 1)
        result = 1 / (1 + np.exp(-z))
        self.assertEqual(mynetwork.MyNetwork.sigmoid(z).all(), result.all())
    def test_vectorized_result(self):
        expected = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0., 0.])
        ld = loader.loader()
        observed = ld.vectorized_result(7)
        self.assertEqual(observed.all(), expected.all())

x = 2
arr1 = np.array([0., 0., 1., 0., 0., 0., 0., 0., 0., 0.])
arr2 = np.zeros(10)
arr2[x] = 1

print(arr1)
print(arr2)

ld = loader.loader()
data = ld.load_data("data/mnist_test.csv")
print(data[0][0])
labels = data[:,0]
data_size = data.shape[0]
data[:,0] = np.zeros((data_size, 10))
print(labels)
print(data[:,0])

if __name__ == "__main__":
    unittest.main()
