#-*-coding:utf-8-*-
'''
Created on Oct 7,2018

@author: pengzhiliang
'''

import numpy as np



def sigmoid(x):
	"""
	sigmoid激活函数
	"""
	return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
	"""
	sigmoid激活函数 的导数
	"""
	return sigmoid(x)*(1-sigmoid(x))

def relu(x):
	"""
	relu激活函数
	"""
	#return 0.5*x + 0.5*abs(x)
	return x*(x>0)

def relu_prime(x):
	"""
	relu 激活函数 的导数
	"""
	#x = relu(x)
	#x[x>0] = 1
	#return x
	return (x>0).astype(x.dtype)
	

def tanh(x):
	"""
	tanh激活函数
	"""
	return 2*sigmoid(2*x) - 1

def tanh_prime(x):
	"""
	tanh激活函数 导数
	"""	
	return 1 - tanh(x)**2

def leaky_relu(x,leak=0.2):
	"""
	Leaky ReLu激活函数 
	"""		
	f1 = 0.5 * (1 + leak)
	f2 = 0.5 * (1 - leak)
	return f1 * x + f2 * abs(x)

def leaky_relu_prime(x,leak=0.2):
	"""
	Leaky ReLu激活函数 导数
	"""	
	dx = np.ones_like(x)
	dx[x < 0] = leak
	return dx

def active_fun(s,x):
	"""
	整合各种激活函数，方便后续使用
	"""
	if s == "sigmoid":
		return sigmoid(x)
	elif s == "relu":
		return relu(x)
	elif s == "tanh":
		return tanh(x)
	elif s == "leaky_relu":
		return leaky_relu(x)
	else:
		raise NameError,"没有此激活函数:%s,待后续添加"%s

def active_fun_prime(s,x):
	"""
	整合各种激活函数 的导数，方便后续使用
	"""
	if s == "sigmoid":
		return sigmoid_prime(x)
	elif s == "relu":
		return relu_prime(x)
	elif s == "tanh":
		return tanh_prime(x)
	elif s == "leaky_relu":
		return leaky_relu_prime(x)
	else:
		pass