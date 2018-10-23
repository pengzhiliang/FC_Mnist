#-*-coding:utf-8-*-
'''
Created on Oct 7,2018

@author: pengzhiliang
@version: 1.4
'''
import numpy as np
from act_fun import *
from Network import *
from Batch_Normalization import *
class network_bn(object):
	"""
	利用BP实现mnist数据集的分类,全连接神经网络,输入层数和各层激活函数均可指定
	"""
	def __init__(self, sizes, activation_fun):
		"""
		size: 例如[784,30,10]
		表示输入层有784个神经元，第一个隐藏层有30个神经元，输出层有10个神经元
		activation_fun：例如["sigmoid","relu",...]
		表示第一层激活函数为sigmoid,第二层为relu,...
		新增特性：
		- 加入Batch Normalization
		"""
		self.sizes = sizes
		self.nn_len = len(sizes)
		self.activation_fun = activation_fun
		# 除去输入层，随机产生每层中 y 个神经元的 biase 值
		# (30,1) (10,1)
		self.biases = [np.random.uniform(0,1,(1,y)) for y in sizes[1:]]
		# 随机产生每层的 weight 值)
		# msra初始化
		self.weights = [np.random.normal(0,np.sqrt(2./sizes[0]),(x, y)) for x, y in zip(sizes[:-1], sizes[1:])]
		
		## BN的初始化
		self.bn_beta = [np.zeros(layer_output_dim) for layer_output_dim in sizes[1:]]
		self.bn_gamma = [np.ones(layer_output_dim) for layer_output_dim in sizes[1:]]
		self.bn_params = []
		self.bn_params = [{'mode': 'train'} for i in xrange(self.nn_len - 1)]
	
	def __call__(self):
		print "初始化完成：\n"
		w = [x.shape for x in self.weights]
		print "权重shape:",w
		b = [x.shape for x in self.biases]
		print "偏置shape:",b
		beta = [x.shape for x in self.bn_beta]
		print "BN beta shape:",beta
		gamma = [x.shape for x in self.bn_gamma]
		print "BN beta shape:",gamma
		print "激活函数：", self.activation_fun	

	def feedforward(self,y):
		"""
		前向传播，计算每层神经元输出
		"""
		# 更新BN mode参数
		mode = 'test'
		for bn_param in self.bn_params:
			bn_param['mode'] = mode

		for w,b,s,beta,gamma,bn_param in zip(self.weights,self.biases,self.activation_fun,self.bn_beta,self.bn_gamma,self.bn_params):
			z = np.dot(y,w) + b[0] #广播机制导致bias维度发生变化，取其中一行即可
			# 加入BN前向传播
			z , cache = batchnorm_forward(z, gamma, beta, bn_param)
			y = active_fun(s , z) # (*,784) * (784,30) = (*,30)
		return y

	def fit(self,X_train,X_test,Y_train,Y_test,epochs,batch_size,learning_rate,display_epochs,decay):
		"""
		采取 小批量随机梯度下降( Mini-batch gradient descent) 进行训练
		X_train	: 训练集
		X_test	: 测试积
		Y_train	: 训练集标签
		Y_test	: 测试集标签
		epochs 	: 迭代次数
		batch_size	: 批数量，默认为32
		learning_rate : 学习速率
		display_epochs :每几轮显示一次测试结果
		decay 	: 学习速率衰减因子 
		"""
		n_test = len(X_test)
		n_train = len(X_train)
		acc = []
		for j in xrange(epochs):
			# 学习速率随迭代次数衰减,实现自适应调节的功能
			learning_rate *= 1./(1+ decay*j)
			for k in xrange(0,n_train,batch_size):
				# 因为训练集已经打乱，可看做随机抽取
				# 按照小样本数量划分训练集,最后会产生一定的浪费，不过无所谓啦
				X = X_train[k : k + batch_size]
				Y = Y_train[k : k + batch_size]
				self.backprop(X,Y,learning_rate)
			nt = self.evaluate(X_test,Y_test)
			acc.append(nt/float(n_test))
			if j % display_epochs == 0:
				print "Epoch {0}: Acc: {1} / {2} = {3}".format(j,nt,n_test,acc[-1])
				#print [beta for beta in self.bn_beta]
		return acc


	def backprop(self,X,Y,learning_rate):
		"""
		计算反向传播的梯度
		"""
		# 更新BN mode参数
		mode = 'train'
		for bn_param in self.bn_params:
			bn_param['mode'] = mode

		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		# 更新BN层beta和gamma参数
		nabla_beta = [np.zeros(beta.shape) for beta in self.bn_beta]
		nabla_gamma = [np.zeros(gamma.shape) for gamma in self.bn_gamma]

		# 前向传输
		activation = X
		# 储存每层的神经元的值的矩阵，下面循环会 append 每层的神经元的值
		activations = [X]
		# 储存每个未经过 激活函数 计算的神经元的值
		zs = []
		# 储存BN层前向传播返回值
		caches = []
		for s,w,b,beta,gamma,bn_param in zip(self.activation_fun,self.weights,self.biases,self.bn_beta,self.bn_gamma,self.bn_params):
			# 当前层的输入，未经过激活函数的值
			z = np.dot(activation,w) + b 
			#加入BN
			z , cache = batchnorm_forward(z, gamma, beta, bn_param)
			# 储存在 zs caches 中
			zs.append(z)
			caches.append(cache)

			# 当前层的输出，已经过激活函数，即为下一层的输入
			activation = active_fun(s,z)
			# 储存在 activations 中
			activations.append(activation) 
			# activations[-1] 即为输出层结果

		# 求 δ 的值
		# delta 对应于 西瓜书p103 (5.10)
		delta = self.cost_derivative(Y,activations[-1]) * active_fun_prime(self.activation_fun[-1],zs[-1])
		# 更新BN层最后一层beta和gamma参数
		delta, dgamma, dbeta = batchnorm_backward(delta,caches[-1])
		nabla_gamma[-1] = dgamma
		nabla_beta[-1] = dbeta
		#用于求隐含层与输出层之间的偏差b 对应于 西瓜书p103 (5.12)
		nabla_b[-1] = -delta
		# 对应于 西瓜书p103 (5.11)
		nabla_w[-1] = np.dot(activations[-2].T,delta)

		# 从后往前计算梯度
		for l in xrange(2, self.nn_len): # -2表示倒数第二层
			# 从倒数第 l 层开始更新，-l 是 python 中特有的语法表示从倒数第 l 层开始计算
			# 倒数第l层的输入，未经过激活函数（但经过wx+b）
			cache = caches[-l]
			z = zs[-l]
			prime = active_fun_prime(self.activation_fun[-l],z)
			# 对应于 西瓜书p104 (5.15)
			delta = np.dot(delta,self.weights[-l+1].T) * prime
			delta, dgamma, dbeta = batchnorm_backward(delta,cache)
			nabla_gamma[-l] = dgamma
			nabla_beta[-l] = dbeta
			nabla_b[-l] = -delta
			# 对应于 西瓜书p103 (5.13)
			nabla_w[-l] = np.dot(activations[-l-1].T,delta)

		# 更新根据累加的偏导值更新 w 和 b，这里因为用了小样本
		self.weights = [w + learning_rate*nw for w, nw in zip(self.weights, nabla_w)]
		self.biases =  [b + learning_rate*nb for b, nb in zip(self.biases, nabla_b)]
		self.bn_beta =  [b + learning_rate*nb for b, nb in zip(self.bn_beta, nabla_beta)]
		self.bn_gamma =  [g + learning_rate*ng for g, ng in zip(self.bn_gamma, nabla_gamma)]

	def evaluate(self, X, Y):
		"""
		评估性能
		"""
		# 获得预测结果
		Y_pre = vectorized_label(np.argmax(self.feedforward(X),axis = 1))
		assert Y.shape == Y_pre.shape , "Y and Y_pre shape 不匹配！！！"
		n = 0
		# 返回正确识别的个数
		for i in np.abs(Y_pre-Y) :
			if sum(i) == 0:
				n += 1
		return n
	def cost_derivative(self, y, output):
		"""
		交叉熵损失函数 导数
		"""
		return ( y - output)	


if __name__ == "__main__":
	nn = network_bn([784,256,128,64,32,10],["relu","relu","relu","relu","sigmoid"])
	# [784,128,32,10],["leaky_relu","sigmoid","sigmoid"] # best 0.9766
	# [784,128,64,32,10],["leaky_relu","leaky_relu","leaky_relu","sigmoid"] # 0.9695
	nn()
	epochs = 200
	X_train,X_test,Y_train,Y_test = load_mnist()
	Acc = nn.fit(X_train=X_train,X_test=X_test,Y_train=Y_train,Y_test=Y_test,
		epochs=epochs,batch_size=32,learning_rate=0.001,display_epochs=1,decay=1e-5)
	plot_acc(Acc,epochs=epochs)
