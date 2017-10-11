import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)


  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  
  for i in xrange(X.shape[0]):
  	z = np.zeros(W.shape[1])
  	for j in xrange(W.shape[1]):
  		for k in xrange(X.shape[1]):
  			z[j] += X[i][k]*W[k][j]
  	max_ind = np.argmax(z)
  	z = z - np.max(z)
  	z = np.exp(z)
  	z = z/np.sum(z)
  	loss += -np.log(z[y[i]])
  	z[y[i]] = z[y[i]] - 1
  	for j in xrange(W.shape[1]):
  		for k in xrange(X.shape[1]):
  			dW[k][j] += X[i][k]*z[j]
  loss = loss/X.shape[0]
  dW = dW/X.shape[0]
  for j in xrange(W.shape[1]):
	for k in xrange(X.shape[1]):
		loss += 0.5*reg*W[k][j]*W[k][j]
		dW[k][j] += reg*W[k][j]
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  z = np.dot(X,W)
  z = z.T - np.max(z,axis=1)
  z = z.T
  z = np.exp(z)
  z = z/np.expand_dims(np.sum(z,axis=1),axis=1)
  loss = np.sum(-np.log(z[range(X.shape[0]),y]))
  loss = loss/X.shape[0]
  loss += 0.5*reg*np.sum(W*W)
  # z = z.T
  grad_1 = np.zeros_like(z)
  grad_1[range(X.shape[0]),y] = 1 
  z = z - grad_1
  dW = np.dot(X.T,z)
  # print dW.shape
  dW = dW/X.shape[0]
  dW += reg*W
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

