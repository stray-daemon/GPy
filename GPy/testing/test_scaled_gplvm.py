
import unittest
import numpy as np
import GPy

class TestScaledGPLVM(unittest.TestCase):
    def test_init(self):
        """
        Test initialization of ScaledGPLVM model
        """
        N, D, Q = 10, 5, 2
        X = np.random.randn(N, Q)
        Y = np.random.randn(N, D)
        
        model = GPy.models.ScaledGPLVM(Y, Q, X=X)
        
        self.assertEqual(model.num_data, N)
        self.assertEqual(model.output_dim, D)
        self.assertEqual(model.input_dim, Q)
        
        self.assertEqual(model.X.shape, (N, Q))
        self.assertEqual(model.scales.shape, (D,))
        
    def test_predictions(self):
        """
        Test predictions of ScaledGPLVM model
        """
        N, D, Q = 10, 5, 2
        X = np.random.randn(N, Q)
        Y = np.random.randn(N, D)
        
        model = GPy.models.ScaledGPLVM(Y, Q, X=X)
        
        Xtest = np.random.randn(5, Q)
        mu, var = model.predict(Xtest)
        
        self.assertEqual(mu.shape, (5, D))
        self.assertEqual(var.shape, (5, D))
        
    def test_optimization(self):
        """
        Test optimization of ScaledGPLVM model
        """
        N, D, Q = 10, 5, 2
        X = np.random.randn(N, Q)
        Y = np.random.randn(N, D)
        
        model = GPy.models.ScaledGPLVM(Y, Q, X=X)
        
        model.optimize(messages=False, max_iters=5)
        
        self.assertEqual(model.scales.shape, (D,))
        
if __name__ == "__main__":
    unittest.main()
