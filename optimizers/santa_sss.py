import math

import numpy

from chainer import cuda
from chainer import optimizer

_default_hyperparam = optimizer.Hyperparameter()
_default_hyperparam.lr = 0.0002
_default_hyperparam.sigma = 0.9
_default_hyperparam.lamb = 1e-8
_default_hyperparam.burnin = 1e+3
_default_hyperparam.gamma = 0.5
_default_hyperparam.A = 1
_default_hyperparam.C = 1
_default_hyperparam.N = 1e+4

class SantaSSSRule(optimizer.UpdateRule):

	'''Update rule of Santa-SSS optimization algorithm.

	Args:
        parent_hyperparam (~chainer.optimizer.Hyperparameter): Hyperparameter
            that provides the default values.
        lr (float): Learning rate.
        sigma (float): Exponential decay rate of the second order moment.
        lamb (float): Small value for the numerical stability.
		burnin (int): Burnin span for MCMC.
		gamma (float): Annealing coefficient for beta.
		A (float): Annealign coefficient for beta.
		C (float): Initial rate of exploration parameter alpha.
		N (int): Total iteration number.

	'''

	def __init__(self, parent_hyperparam = None,
                 lr = None, sigma = None, lamb = None, burnin = None,
				 gamma = None, A = None, C = None, N = None):
        super(SantaSSSRule, self).__init__(
            parent_hyperparam or _default_hyperparam)
        if lr is not None:
            self.hyperparam.eta = lr
        if sigma is not None:
            self.hyperparam.sigma = sigma
        if lamb is not None:
            self.hyperparam.lamb = lamb
        if burnin is not None:
            self.hyperparam.eps = burnin
		if gamma is not None:
            self.hyperparam.gamma = gamma
        if A is not None:
            self.hyperparam.A = A
        if C is not None:
            self.hyperparam.C = C
		if N is not None:
			self.hyperparam.N = N

	def init_state(self, param):
        xp = cuda.get_array_module(param.data)
        with cuda.get_device_from_array(param.data):
			hp = self.hyperparam
            self.state['v'] = xp.zeros_like(param.data)
            self.state['alpha'] = hp.C * xp.sqrt(hp.lr) * xp.ones_like(param.data)
			self.state['u'] = xp.random.randn(param.data.shape)
			self.state['beta'] = hp.A
			self.state['g'] = xp.zeros_like(param.data)
			self.state['count'] = 0

	def update_core_cpu(self, param):
        grad = param.grad
        if grad is None:
            return
        hp = self.hyperparam
        lamb = grad.dtype.type(hp.lamb)
        if hp.lamb != 0 and lamb == 0:
            raise ValueError(
                'lamb of Santa-SSS optimizer is too small for {} ({})'.format(
                    grad.dtype.name, hp.lamb))
        v, alpha = self.state['v'], self.state['alpha']
		bu, beta = self.state['u'], self.state['beta']
		bg, count = self.state['g'], self.state['count']

		v *= hp.sigma
        v += (1 - hp.sigma) / (hp.N * hp.N) * grad * grad
		g = 1 / numpy.sqrt(hp.lamb + numpy.sqrt(v))
		param.data += g * bu / 2

        if count < hp.burnin:
			# exploration
