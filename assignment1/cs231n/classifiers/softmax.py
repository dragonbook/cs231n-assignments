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
  #pass
  num_classes = W.shape[1]
  num_train = X.shape[0]

  for i in xrange(num_train):
    scores = X[i].dot(W)
    scores -= np.max(scores)
    scores_exp = np.exp(scores)
    sum_exp = np.sum(scores_exp)
    prob = scores_exp[y[i]] / sum_exp
    loss += -np.log(prob)

    dscores = -1.0/prob * (-scores_exp[y[i]]/(sum_exp**2)) * scores_exp
    dscores[y[i]] += -1.0/prob * (1.0/sum_exp) * scores_exp[y[i]]

    #print('X[i] shape: ', X[i].shape)
    #print('dscores shape: ', dscores.shape)

    dW += np.dot(X[i].reshape(-1, 1), dscores.reshape(1, -1))

  loss /= num_train
  dW /= num_train

  loss += reg * np.sum(W**2)
  dW += W*reg*2

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
  #pass
  scores = np.dot(X, W)
  scores -= np.max(scores, axis=1).reshape(-1, 1)
  scores_exp = np.exp(scores)
  scores_exp_y = scores_exp[np.arange(X.shape[0]), y]
  sum_exp = np.sum(scores_exp, axis=1)
  prob = scores_exp_y / sum_exp
  loss = np.sum(-np.log(prob)) / X.shape[0]
  loss += np.sum(W*W) * reg

  dscores = np.reshape(-1.0/prob * (-scores_exp_y / (sum_exp**2)), (-1, 1)) * scores_exp
  dscores[np.arange(X.shape[0]), y] += -1.0/prob * (1.0/sum_exp) * scores_exp_y
  dW = np.dot(X.T, dscores) / X.shape[0] + W*reg*2
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
