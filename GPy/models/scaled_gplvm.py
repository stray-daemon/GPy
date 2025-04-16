
import numpy as np
from ..core import Model
from .. import likelihoods
from .. import kern
from ..core.parameterization.variational import NormalPosterior
from ..core.parameterization import Param
from paramz.transformations import Logexp
from ..util.initialization import initialize_latent

class ScaledGPLVM(Model):
    """
    Gaussian Process Latent Variable Model with scaling parameters for each output dimension
    
    :param Y: observed data (np.ndarray) - N x D
    :type Y: np.ndarray
    :param input_dim: latent dimensionality
    :type input_dim: int
    :param init: initialisation method for the latent space
    :type init: 'PCA'|'random'
    :param kernel: the kernel to use for the GP
    :type kernel: a GPy kernel
    :param X: initial latent space
    :type X: np.ndarray or None
    :param sigma: standard deviation of the Gaussian noise
    :type sigma: float
    """
    def __init__(self, Y, input_dim, X=None, kernel=None, init='PCA', sigma=1.0, name='scaled_gplvm'):
        super(ScaledGPLVM, self).__init__(name=name)
        
        self.Y = Y
        self.num_data, self.output_dim = Y.shape
        self.input_dim = input_dim
        
        if X is None:
            X, fracs = initialize_latent(init, input_dim, Y)
        else:
            assert X.shape[0] == self.num_data
            assert X.shape[1] == self.input_dim
        
        if kernel is None:
            kernel = kern.RBF(input_dim, ARD=True)
        
        self.likelihood = likelihoods.Gaussian(variance=sigma**2)
        
        self.X = Param('latent_mean', X)
        
        self.scales = Param('scales', np.ones(self.output_dim), Logexp())
        
        self.link_parameter(self.X, self.scales)
        self.kern = kernel
        self.link_parameter(self.kern)
        
        self.variational_prior = NormalPosterior(means=X, variances=np.ones((self.num_data, input_dim)))
        
    def parameters_changed(self):
        """
        Update model parameters
        """
        Y_scaled = self.Y / self.scales[None, :]
        
        K = self.kern.K(self.X)
        
        K_noisy = K + np.eye(self.num_data) * self.likelihood.variance
        
        _log_marginal = 0
        
        for d in range(self.output_dim):
            Wi, LW, LWi, W_logdet = self._get_matrices(K_noisy)
            _log_marginal += self._log_likelihood_dim(Y_scaled[:, d:d+1], Wi, W_logdet)
            
        self._log_marginal = _log_marginal
        
        self._update_gradients(Y_scaled, K, K_noisy)
        
    def _get_matrices(self, K_noisy):
        """
        Get matrices needed for likelihood computation
        """
        Wi, LW, LWi, W_logdet = np.linalg.inv(K_noisy), np.linalg.cholesky(K_noisy), np.linalg.cholesky(np.linalg.inv(K_noisy)), np.linalg.slogdet(K_noisy)[1]
        return Wi, LW, LWi, W_logdet
    
    def _log_likelihood_dim(self, Y_d, Wi, W_logdet):
        """
        Compute log likelihood for a single output dimension
        """
        return -0.5 * (np.dot(Y_d.T, np.dot(Wi, Y_d)).sum() + W_logdet + self.num_data * np.log(2 * np.pi))
    
    def _update_gradients(self, Y_scaled, K, K_noisy):
        """
        Update gradients of the model parameters
        """
        dL_dK = np.zeros_like(K)
        Wi = np.linalg.inv(K_noisy)
        
        for d in range(self.output_dim):
            dL_dK += 0.5 * (np.dot(Wi, np.dot(Y_scaled[:, d:d+1], Y_scaled[:, d:d+1].T)).dot(Wi) - Wi)
        
        self.kern.update_gradients_full(dL_dK, self.X)
        
        self.X.gradient = self.kern.gradients_X(dL_dK, self.X)
        
        self.scales.gradient = np.zeros(self.output_dim)
        for d in range(self.output_dim):
            y_d = self.Y[:, d:d+1]
            y_scaled_d = Y_scaled[:, d:d+1]
            self.scales.gradient[d] = -np.dot(y_scaled_d.T, np.dot(Wi, y_scaled_d)).sum() / self.scales[d]
    
    def predict(self, Xnew):
        """
        Make predictions at new points
        
        :param Xnew: new points for prediction
        :type Xnew: np.ndarray
        """
        Kx = self.kern.K(self.X, Xnew)
        Kxx = self.kern.K(Xnew)
        
        K_noisy = self.kern.K(self.X) + np.eye(self.num_data) * self.likelihood.variance
        Wi = np.linalg.inv(K_noisy)
        
        mu = np.dot(Kx.T, np.dot(Wi, self.Y))
        
        mu = mu * self.scales[None, :]
        
        var = Kxx - np.dot(Kx.T, np.dot(Wi, Kx))
        var = var[:, None] * (self.scales[None, :]**2)
        
        return mu, var
