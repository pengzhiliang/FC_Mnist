#-*-coding:utf-8-*-
'''
Created on Oct 8,2018

@author: pengzhiliang
@reference: https://blog.csdn.net/qunnie_yi/article/details/80128445
'''
import numpy as np

def batchnorm_forward(x, gamma, beta, bn_param):
	"""
	输入:
	- x: 输入数据 shape (N, D)
	- gamma: 缩放参数 shape (D,)
	- beta: 平移参数 shape (D,)
	- bn_param: 包含如下参数的dict:
	- mode: 'train' or 'test'; 用来区分训练还是测试
	- eps: 除以方差时为了防止方差太小而导致数值计算不稳定
	- momentum: 前面讨论的momentum.
	- running_mean: 数组 shape (D,) 记录最新的均值
	- running_var 数组 shape (D,) 记录最新的方差
	
	返回一个tuple:
	- out: shape (N, D)
	- cache: 缓存反向计算时需要的变量
	 """
	mode = bn_param['mode']
	eps = bn_param.get('eps', 1e-5)
	momentum = bn_param.get('momentum', 0.9)
 
	N, D = x.shape
	running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
	running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))
 
	out, cache = None, None
	if mode == 'train':

		sample_mean = np.mean(x, axis = 0) 
		sample_var = np.var(x, axis = 0)
 
		x_normalized = (x-sample_mean) / np.sqrt(sample_var + eps)
		out = gamma*x_normalized + beta
 
		running_mean = momentum * running_mean + (1 - momentum) * sample_mean
		running_var = momentum * running_var + (1 - momentum) * sample_var
 
		cache = (x, sample_mean, sample_var, x_normalized, beta, gamma, eps)

	elif mode == 'test':
    
		x_normalized = (x - running_mean)/np.sqrt(running_var +eps)
		out = gamma*x_normalized + beta

	else:
		raise ValueError('Invalid forward batchnorm mode "%s"' % mode)
 
	# Store the updated running means back into bn_param
	bn_param['running_mean'] = running_mean
	bn_param['running_var'] = running_var
 
	return out, cache

def batchnorm_backward(dout, cache):
	""" 
	Inputs:
	- dout: Upstream derivatives, of shape (N, D)
	- cache: Variable of intermediates from batchnorm_forward.
	
	Returns a tuple of:
	- dx: Gradient with respect to inputs x, of shape (N, D)
	- dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
	- dbeta: Gradient with respect to shift parameter beta, of shape (D,)
	"""
	dx, dgamma, dbeta = None, None, None

	(x, sample_mean, sample_var, x_normalized, beta, gamma, eps) = cache
	N = x.shape[0]
	dbeta = np.sum(dout, axis=0)
	dgamma = np.sum(x_normalized*dout, axis = 0)
	dx_normalized = gamma* dout
	dsample_var = np.sum(-1.0/2*dx_normalized*(x-sample_mean)/(sample_var+eps)**(3.0/2), axis =0)
	dsample_mean = np.sum(-1/np.sqrt(sample_var+eps)* dx_normalized, axis = 0) + 1.0/N*dsample_var *np.sum(-2*(x-sample_mean), axis = 0) 
	dx = 1/np.sqrt(sample_var+eps)*dx_normalized + dsample_var*2.0/N*(x-sample_mean) + 1.0/N*dsample_mean

	return dx, dgamma, dbeta