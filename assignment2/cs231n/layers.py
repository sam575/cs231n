from builtins import range
import numpy as np


def affine_forward(x, w, b):
	"""
	Computes the forward pass for an affine (fully-connected) layer.

	The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
	examples, where each example x[i] has shape (d_1, ..., d_k). We will
	reshape each input into a vector of dimension D = d_1 * ... * d_k, and
	then transform it to an output vector of dimension M.

	Inputs:
	- x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
	- w: A numpy array of weights, of shape (D, M)
	- b: A numpy array of biases, of shape (M,)

	Returns a tuple of:
	- out: output, of shape (N, M)
	- cache: (x, w, b)
	"""
	out = None
	###########################################################################
	# TODO: Implement the affine forward pass. Store the result in out. You   #
	# will need to reshape the input into rows.                               #
	###########################################################################
	x_reshape = x.reshape(x.shape[0],-1)
	out = np.dot(x_reshape,w)+np.expand_dims(b,0) 
	pass
	###########################################################################
	#                             END OF YOUR CODE                            #
	###########################################################################
	cache = (x, w, b)
	return out, cache


def affine_backward(dout, cache):
	"""
	Computes the backward pass for an affine layer.

	Inputs:
	- dout: Upstream derivative, of shape (N, M)
	- cache: Tuple of:
	  - x: Input data, of shape (N, d_1, ... d_k)
	  - w: Weights, of shape (D, M)

	Returns a tuple of:
	- dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
	- dw: Gradient with respect to w, of shape (D, M)
	- db: Gradient with respect to b, of shape (M,)
	"""
	x, w, b = cache
	dx, dw, db = None, None, None
	###########################################################################
	# TODO: Implement the affine backward pass.                               #
	###########################################################################
	db = np.sum(dout,0)
	x_reshape = x.reshape(x.shape[0],-1)
	dx = np.dot(dout,w.T)
	dx = np.reshape(dx,x.shape)
	dw = np.dot((x_reshape).T,dout)
	pass
	###########################################################################
	#                             END OF YOUR CODE                            #
	###########################################################################
	return dx, dw, db


def relu_forward(x):
	"""
	Computes the forward pass for a layer of rectified linear units (ReLUs).

	Input:
	- x: Inputs, of any shape

	Returns a tuple of:
	- out: Output, of the same shape as x
	- cache: x
	"""
	out = None
	###########################################################################
	# TODO: Implement the ReLU forward pass.                                  #
	###########################################################################
	out = np.clip(0,x,None)
	pass
	###########################################################################
	#                             END OF YOUR CODE                            #
	###########################################################################
	cache = x
	return out, cache


def relu_backward(dout, cache):
	"""
	Computes the backward pass for a layer of rectified linear units (ReLUs).

	Input:
	- dout: Upstream derivatives, of any shape
	- cache: Input x, of same shape as dout

	Returns:
	- dx: Gradient with respect to x
	"""
	dx, x = None, cache
	###########################################################################
	# TODO: Implement the ReLU backward pass.                                 #
	###########################################################################
	dx = np.clip(x,0,None)
	dx[np.nonzero(dx)] = 1
	dx *= dout
	pass
	###########################################################################
	#                             END OF YOUR CODE                            #
	###########################################################################
	return dx


def batchnorm_forward(x, gamma, beta, bn_param):
	"""
	Forward pass for batch normalization.

	During training the sample mean and (uncorrected) sample variance are
	computed from minibatch statistics and used to normalize the incoming data.
	During training we also keep an exponentially decaying running mean of the
	mean and variance of each feature, and these averages are used to normalize
	data at test-time.

	At each timestep we update the running averages for mean and variance using
	an exponential decay based on the momentum parameter:

	running_mean = momentum * running_mean + (1 - momentum) * sample_mean
	running_var = momentum * running_var + (1 - momentum) * sample_var

	Note that the batch normalization paper suggests a different test-time
	behavior: they compute sample mean and variance for each feature using a
	large number of training images rather than using a running average. For
	this implementation we have chosen to use running averages instead since
	they do not require an additional estimation step; the torch7
	implementation of batch normalization also uses running averages.

	Input:
	- x: Data of shape (N, D)
	- gamma: Scale parameter of shape (D,)
	- beta: Shift paremeter of shape (D,)
	- bn_param: Dictionary with the following keys:
	  - mode: 'train' or 'test'; required
	  - eps: Constant for numeric stability
	  - momentum: Constant for running mean / variance.
	  - running_mean: Array of shape (D,) giving running mean of features
	  - running_var Array of shape (D,) giving running variance of features

	Returns a tuple of:
	- out: of shape (N, D)
	- cache: A tuple of values needed in the backward pass
	"""
	mode = bn_param['mode']
	eps = bn_param.get('eps', 1e-5)
	momentum = bn_param.get('momentum', 0.9)

	N, D = x.shape
	running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
	running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

	out, cache = None, None
	if mode == 'train':
		#######################################################################
		# TODO: Implement the training-time forward pass for batch norm.      #
		# Use minibatch statistics to compute the mean and variance, use      #
		# these statistics to normalize the incoming data, and scale and      #
		# shift the normalized data using gamma and beta.                     #
		#                                                                     #
		# You should store the output in the variable out. Any intermediates  #
		# that you need for the backward pass should be stored in the cache   #
		# variable.                                                           #
		#                                                                     #
		# You should also use your computed sample mean and variance together #
		# with the momentum variable to update the running mean and running   #
		# variance, storing your result in the running_mean and running_var   #
		# variables.                                                          #
		#######################################################################
		x_mean = np.mean(x,axis=0)
		x_var = np.sqrt(np.var(x,axis=0)+eps)
		x_cap = (x-x_mean)/x_var
		out = gamma*(x_cap)+beta
		running_mean = momentum*running_mean + (1-momentum)*x_mean
		running_var = momentum*running_var + (1-momentum)*np.var(x,axis=0)
		cache = (x_cap,x_mean,x_var,eps,gamma)
		pass
		#######################################################################
		#                           END OF YOUR CODE                          #
		#######################################################################
	elif mode == 'test':
		#######################################################################
		# TODO: Implement the test-time forward pass for batch normalization. #
		# Use the running mean and variance to normalize the incoming data,   #
		# then scale and shift the normalized data using gamma and beta.      #
		# Store the result in the out variable.                               #
		#######################################################################
		out = gamma*((x-running_mean)/np.sqrt(running_var+eps)) + beta
		pass
		#######################################################################
		#                          END OF YOUR CODE                           #
		#######################################################################
	else:
		raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

	# Store the updated running means back into bn_param
	bn_param['running_mean'] = running_mean
	bn_param['running_var'] = running_var

	return out, cache


def batchnorm_backward(dout, cache):
	"""
	Backward pass for batch normalization.

	For this implementation, you should write out a computation graph for
	batch normalization on paper and propagate gradients backward through
	intermediate nodes.

	Inputs:
	- dout: Upstream derivatives, of shape (N, D)
	- cache: Variable of intermediates from batchnorm_forward.

	Returns a tuple of:
	- dx: Gradient with respect to inputs x, of shape (N, D)
	- dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
	- dbeta: Gradient with respect to shift parameter beta, of shape (D,)
	"""
	dx, dgamma, dbeta = None, None, None
	###########################################################################
	# TODO: Implement the backward pass for batch normalization. Store the    #
	# results in the dx, dgamma, and dbeta variables.                         #
	###########################################################################
	N = dout.shape[0]
	x_cap,x_mean,x_var,eps,gamma = cache
	dbeta = np.sum(dout,axis=0)
	dgamma = np.sum(dout*x_cap,axis=0)
	dx = N*dout - np.sum(dout,axis=0) - x_cap*np.sum(x_cap*dout,axis=0)
	dx *= gamma/(N*x_var)
	# print dx.shape
	pass
	###########################################################################
	#                             END OF YOUR CODE                            #
	###########################################################################

	return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
	"""
	Alternative backward pass for batch normalization.

	For this implementation you should work out the derivatives for the batch
	normalizaton backward pass on paper and simplify as much as possible. You
	should be able to derive a simple expression for the backward pass.

	Note: This implementation should expect to receive the same cache variable
	as batchnorm_backward, but might not use all of the values in the cache.

	Inputs / outputs: Same as batchnorm_backward
	"""
	dx, dgamma, dbeta = None, None, None
	###########################################################################
	# TODO: Implement the backward pass for batch normalization. Store the    #
	# results in the dx, dgamma, and dbeta variables.                         #
	#                                                                         #
	# After computing the gradient with respect to the centered inputs, you   #
	# should be able to compute gradients with respect to the inputs in a     #
	# single statement; our implementation fits on a single 80-character line.#
	###########################################################################
	pass
	###########################################################################
	#                             END OF YOUR CODE                            #
	###########################################################################

	return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
	"""
	Performs the forward pass for (inverted) dropout.

	Inputs:
	- x: Input data, of any shape
	- dropout_param: A dictionary with the following keys:
	  - p: Dropout parameter. We drop each neuron output with probability p.
	  - mode: 'test' or 'train'. If the mode is train, then perform dropout;
		if the mode is test, then just return the input.
	  - seed: Seed for the random number generator. Passing seed makes this
		function deterministic, which is needed for gradient checking but not
		in real networks.

	Outputs:
	- out: Array of the same shape as x.
	- cache: tuple (dropout_param, mask). In training mode, mask is the dropout
	  mask that was used to multiply the input; in test mode, mask is None.
	"""
	p, mode = dropout_param['p'], dropout_param['mode']
	if 'seed' in dropout_param:
		np.random.seed(dropout_param['seed'])

	mask = None
	out = None

	if mode == 'train':
		#######################################################################
		# TODO: Implement training phase forward pass for inverted dropout.   #
		# Store the dropout mask in the mask variable.                        #
		#######################################################################
		# mask = np.ones(x.shape)
		# mask.ravel()[np.random.choice(x.size,int(p*x.size))] = 0
		mask = np.random.random_sample(x.shape)
		# mask[mask<=p] = 0.0
		# mask[mask>p] = 1.0
		mask = mask >= p
		out = (x*mask)/(1-p)
		pass
		#######################################################################
		#                           END OF YOUR CODE                          #
		#######################################################################
	elif mode == 'test':
		#######################################################################
		# TODO: Implement the test phase forward pass for inverted dropout.   #
		#######################################################################
		out = x
		pass
		#######################################################################
		#                            END OF YOUR CODE                         #
		#######################################################################

	cache = (dropout_param, mask)
	out = out.astype(x.dtype, copy=False)

	return out, cache


def dropout_backward(dout, cache):
	"""
	Perform the backward pass for (inverted) dropout.

	Inputs:
	- dout: Upstream derivatives, of any shape
	- cache: (dropout_param, mask) from dropout_forward.
	"""
	dropout_param, mask = cache
	mode = dropout_param['mode']
	p = dropout_param['p']
	dx = None
	if mode == 'train':
		#######################################################################
		# TODO: Implement training phase backward pass for inverted dropout   #
		#######################################################################
		dx = dout*mask/(1-p)
		pass
		#######################################################################
		#                          END OF YOUR CODE                           #
		#######################################################################
	elif mode == 'test':
		dx = dout
	return dx


def conv_forward_naive(x, w, b, conv_param):
	"""
	A naive implementation of the forward pass for a convolutional layer.

	The input consists of N data points, each with C channels, height H and
	width W. We convolve each input with F different filters, where each filter
	spans all C channels and has height HH and width HH.

	Input:
	- x: Input data of shape (N, C, H, W)
	- w: Filter weights of shape (F, C, HH, WW)
	- b: Biases, of shape (F,)
	- conv_param: A dictionary with the following keys:
	  - 'stride': The number of pixels between adjacent receptive fields in the
		horizontal and vertical directions.
	  - 'pad': The number of pixels that will be used to zero-pad the input.

	Returns a tuple of:
	- out: Output data, of shape (N, F, H', W') where H' and W' are given by
	  H' = 1 + (H + 2 * pad - HH) / stride
	  W' = 1 + (W + 2 * pad - WW) / stride
	- cache: (x, w, b, conv_param)
	"""
	out = None
	###########################################################################
	# TODO: Implement the convolutional forward pass.                         #
	# Hint: you can use the function np.pad for padding.                      #
	###########################################################################
	stride,pad = conv_param['stride'],conv_param['pad']
	# print x.shape,w.shape
	h_ = 1+(x.shape[2]+2*pad - w.shape[2])/stride
	w_ = 1+(x.shape[3]+2*pad - w.shape[3])/stride
	out = np.zeros((x.shape[0],w.shape[0],h_,w_))
	x_pad = np.lib.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)),'constant')
	# print x.shape
	# print out.shape
	for i in xrange(x.shape[0]):
		for c in xrange(out.shape[1]):
			h_s = 0
			for hh in xrange(h_):
				w_s = 0
				h_e = h_s+w.shape[2]
				for ww in xrange(w_):
					w_e = w_s+w.shape[3]
					# x_win = x[i,:,h_s:h_e,w_s:w_e]
					out[i,c,hh,ww] = np.sum(np.multiply(x_pad[i,:,h_s:h_e,w_s:w_e],w[c]))+b[c]
					w_s += stride
				h_s += stride
	pass
	###########################################################################
	#                             END OF YOUR CODE                            #
	###########################################################################
	cache = (x, w, b, conv_param)
	return out, cache


def conv_backward_naive(dout, cache):
	"""
	A naive implementation of the backward pass for a convolutional layer.

	Inputs:
	- dout: Upstream derivatives.
	- cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive
	
	- x: Input data of shape (N, C, H, W)
	- w: Filter weights of shape (F, C, HH, WW)
	- out: Output data, of shape (N, F, H', W') where H' and W' are given by
	
	Returns a tuple of:
	- dx: Gradient with respect to x
	- dw: Gradient with respect to w
	- db: Gradient with respect to b
	"""

	dx, dw, db = None, None, None
	###########################################################################
	# TODO: Implement the convolutional backward pass.                        #
	###########################################################################
	x, w, b, conv_param = cache
	stride,pad = conv_param['stride'],conv_param['pad']
	db = np.sum(dout,axis=(0,2,3))
	dw = np.zeros_like(w)
	dx = np.zeros_like(x)
	x_pad = np.lib.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)),'constant')
	dx_pad = np.lib.pad(dx,((0,0),(0,0),(pad,pad),(pad,pad)),'constant')
	for i in xrange(w.shape[0]):
		for c in xrange(w.shape[1]):
			# h_s = 0
			for hh in xrange(w.shape[2]):
				# w_s = 0
				h_e = hh+dout.shape[2]*stride
				#in dout ind will be 0,1*std,2*stride,3*stride... dout times
				for ww in xrange(w.shape[3]):
					w_e = ww+dout.shape[3]*stride
					h_,w_ = np.meshgrid(np.arange(hh,h_e,stride),np.arange(ww,w_e,stride),indexing='ij')
					# hlol = np.mgrid[np.arange(h_s,h_e,stride),np.arange(w_s,w_e,stride)]				
					# print h_
					# print w_
					# print x[:,c,h_,w_].shape
					dw[i,c,hh,ww] = np.sum(np.multiply(dout[:,i,:,:],x_pad[:,c,h_,w_]))
					# w_s += stride
					# break
					# print 'lol'
				# h_s += stride
				# break
	pad_out = w.shape[2]-pad-1
	dout_pad = np.lib.pad(dout,((0,0),(0,0),(pad_out,pad_out),(pad_out,pad_out)),'constant')
	# print dout_pad.shape
	# w = np.flip(w,axis=2)
	# w = np.flip(w,axis=3)
	# print 'original dx: ',dx.shape
	
	for i in xrange(dout.shape[0]):
		for hh in xrange(dout.shape[2]):
			for ww in xrange(dout.shape[3]):
				dx_pad[i,:,hh*stride:hh*stride+w.shape[2],ww*stride:ww*stride+w.shape[3]] += np.sum(dout[i,:,hh,ww][:,None,None,None]*w,axis=0)
	pass

	# print dw.shape
	###########################################################################
	#                             END OF YOUR CODE                            #
	###########################################################################
	return dx_pad[:,:,pad:-pad,pad:-pad], dw, db


def max_pool_forward_naive(x, pool_param):
	"""
	A naive implementation of the forward pass for a max pooling layer.

	Inputs:
	- x: Input data, of shape (N, C, H, W)
	- pool_param: dictionary with the following keys:
	  - 'pool_height': The height of each pooling region
	  - 'pool_width': The width of each pooling region
	  - 'stride': The distance between adjacent pooling regions

	Returns a tuple of:
	- out: Output data
	- cache: (x, pool_param)
	"""
	out = None
	###########################################################################
	# TODO: Implement the max pooling forward pass                            #
	###########################################################################
	ph,pw,stride = pool_param['pool_height'],pool_param['pool_width'],pool_param['stride']
	h_ = 1+(x.shape[2] - ph)/stride
	w_ = 1+(x.shape[3] - pw)/stride
	out = np.zeros((x.shape[0],x.shape[1],h_,w_))
	for hh in xrange(h_):
		for ww in xrange(w_):
			out[:,:,hh,ww] = np.max(x[:,:,hh*stride:hh*stride+ph,ww*stride:ww*stride+pw],axis=(2,3))

	pass
	###########################################################################
	#                             END OF YOUR CODE                            #
	###########################################################################
	cache = (x, pool_param)
	return out, cache


def max_pool_backward_naive(dout, cache):
	"""
	A naive implementation of the backward pass for a max pooling layer.

	Inputs:
	- dout: Upstream derivatives
	- cache: A tuple of (x, pool_param) as in the forward pass.

	Returns:
	- dx: Gradient with respect to x
	"""
	dx = None
	###########################################################################
	# TODO: Implement the max pooling backward pass                           #
	###########################################################################
	x,pool_param = cache
	ph,pw,stride = pool_param['pool_height'],pool_param['pool_width'],pool_param['stride']
	h_ = 1+(x.shape[2] - ph)/stride
	w_ = 1+(x.shape[3] - pw)/stride
	dx = np.zeros_like(x)
	# dx = x
	for hh in xrange(h_):
		for ww in xrange(w_):
			# out[:,:,hh,ww] = np.max(x[:,:,hh*stride:hh*stride+ph,ww*stride:ww*stride+pw],axis=(2,3))
			x_max = np.argmax(x[:,:,hh*stride:hh*stride+ph,ww*stride:ww*stride+pw].reshape(x.shape[0],x.shape[1],-1),axis=(2))
			# print x_max.shape
			# print x_max
			# print dx[:,:,hh*stride + x_max.ravel()/pw,ww*stride + x_max.ravel()%pw].shape
			for l,k in enumerate(x_max.ravel()):
				dx[l/x.shape[1],l%x.shape[1],hh*stride + k/pw,ww*stride + k%pw] += dout[l/x.shape[1],l%x.shape[1],hh,ww]
			# pass_back = x[:,:,hh*stride:hh*stride+ph,ww*stride:ww*stride+pw]/np.max(x[:,:,hh*stride:hh*stride+ph,ww*stride:ww*stride+pw],axis=(2,3))[:,:,None,None]
			# dx[:,:,hh*stride:hh*stride+ph,ww*stride:ww*stride+pw] += pass_back.astype(np.int32)
			# break
		# break
	pass
	# print dx.shape
	# print dx
	###########################################################################
	#                             END OF YOUR CODE                            #
	###########################################################################
	return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
	"""
	Computes the forward pass for spatial batch normalization.

	Inputs:
	- x: Input data of shape (N, C, H, W)
	- gamma: Scale parameter, of shape (C,)
	- beta: Shift parameter, of shape (C,)
	- bn_param: Dictionary with the following keys:
	  - mode: 'train' or 'test'; required
	  - eps: Constant for numeric stability
	  - momentum: Constant for running mean / variance. momentum=0 means that
		old information is discarded completely at every time step, while
		momentum=1 means that new information is never incorporated. The
		default of momentum=0.9 should work well in most situations.
	  - running_mean: Array of shape (D,) giving running mean of features
	  - running_var Array of shape (D,) giving running variance of features

	Returns a tuple of:
	- out: Output data, of shape (N, C, H, W)
	- cache: Values needed for the backward pass
	"""
	mode = bn_param['mode']
	eps = bn_param.get('eps', 1e-5)
	momentum = bn_param.get('momentum', 0.9)

	N,C,H,W = x.shape
	running_mean = bn_param.get('running_mean', np.zeros(C, dtype=x.dtype))
	running_var = bn_param.get('running_var', np.zeros(C, dtype=x.dtype))

	out, cache = None, None
	if mode == 'train':
		###########################################################################
		# TODO: Implement the forward pass for spatial batch normalization.       #
		#                                                                         #
		# HINT: You can implement spatial batch normalization using the vanilla   #
		# version of batch normalization defined above. Your implementation should#
		# be very short; ours is less than five lines.                            #
		###########################################################################
		x_mean = np.mean(x,axis=(0,2,3))
		x_var = np.sqrt(np.var(x,axis=(0,2,3))+eps)
		x_cap = (x-x_mean[None,:,None,None])/x_var[None,:,None,None]
		out = gamma[:,None,None]*(x_cap)+beta[:,None,None]
		running_mean = momentum*running_mean + (1-momentum)*x_mean
		running_var = momentum*running_var + (1-momentum)*np.var(x,axis=(0,2,3))
		cache = (x_cap,x_mean,x_var,eps,gamma)
		pass
		#######################################################################
		#                           END OF YOUR CODE                          #
		#######################################################################
	elif mode == 'test':
		#######################################################################
		# TODO: Implement the test-time forward pass for batch normalization. #
		# Use the running mean and variance to normalize the incoming data,   #
		# then scale and shift the normalized data using gamma and beta.      #
		# Store the result in the out variable.                               #
		#######################################################################
		out = gamma[:,None,None]*((x-running_mean[:,None,None])/np.sqrt(running_var[:,None,None]+eps)) + beta[:,None,None]
		pass
		#######################################################################
		#                          END OF YOUR CODE                           #
		#######################################################################
	else:
		raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

	# Store the updated running means back into bn_param
	bn_param['running_mean'] = running_mean
	bn_param['running_var'] = running_var

	return out, cache


def spatial_batchnorm_backward(dout, cache):
	"""
	Computes the backward pass for spatial batch normalization.

	Inputs:
	- dout: Upstream derivatives, of shape (N, C, H, W)
	- cache: Values from the forward pass

	Returns a tuple of:
	- dx: Gradient with respect to inputs, of shape (N, C, H, W)
	- dgamma: Gradient with respect to scale parameter, of shape (C,)
	- dbeta: Gradient with respect to shift parameter, of shape (C,)
	"""
	dx, dgamma, dbeta = None, None, None

	###########################################################################
	# TODO: Implement the backward pass for spatial batch normalization.      #
	#                                                                         #
	# HINT: You can implement spatial batch normalization using the vanilla   #
	# version of batch normalization defined above. Your implementation should#
	# be very short; ours is less than five lines.                            #
	###########################################################################
	# N = dout.shape[0]
	N = dout.size/dout.shape[1]
	x_cap,x_mean,x_var,eps,gamma = cache
	dbeta = np.sum(dout,axis=(0,2,3))
	dgamma = np.sum(dout*x_cap,axis=(0,2,3))
	dx = N*dout - np.sum(dout,axis=(0,2,3))[:,None,None] - x_cap*np.sum(x_cap*dout,axis=(0,2,3))[:,None,None]
	dx *= (gamma/(N*x_var))[:,None,None]
	pass
	###########################################################################
	#                             END OF YOUR CODE                            #
	###########################################################################

	return dx, dgamma, dbeta


def svm_loss(x, y):
	"""
	Computes the loss and gradient using for multiclass SVM classification.

	Inputs:
	- x: Input data, of shape (N, C) where x[i, j] is the score for the jth
	  class for the ith input.
	- y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
	  0 <= y[i] < C

	Returns a tuple of:
	- loss: Scalar giving the loss
	- dx: Gradient of the loss with respect to x
	"""
	N = x.shape[0]
	correct_class_scores = x[np.arange(N), y]
	margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
	margins[np.arange(N), y] = 0
	loss = np.sum(margins) / N
	num_pos = np.sum(margins > 0, axis=1)
	dx = np.zeros_like(x)
	dx[margins > 0] = 1
	dx[np.arange(N), y] -= num_pos
	dx /= N
	return loss, dx


def softmax_loss(x, y):
	"""
	Computes the loss and gradient for softmax classification.

	Inputs:
	- x: Input data, of shape (N, C) where x[i, j] is the score for the jth
	  class for the ith input.
	- y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
	  0 <= y[i] < C

	Returns a tuple of:
	- loss: Scalar giving the loss
	- dx: Gradient of the loss with respect to x
	"""
	shifted_logits = x - np.max(x, axis=1, keepdims=True)
	Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
	log_probs = shifted_logits - np.log(Z)
	probs = np.exp(log_probs)
	N = x.shape[0]
	loss = -np.sum(log_probs[np.arange(N), y]) / N
	dx = probs.copy()
	dx[np.arange(N), y] -= 1
	dx /= N
	return loss, dx