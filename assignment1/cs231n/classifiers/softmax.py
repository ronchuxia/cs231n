from builtins import range
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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]

    for i in range(num_train):
        scores = np.zeros(num_classes)
        for j in range(num_classes):
            score = X[i] @ W[:, j]
            scores[j] = score
        prob = np.exp(scores - np.max(scores)) / np.sum(np.exp(scores - np.max(scores)))
        loss += - np.log(prob[y[i]])

        d_scores = prob 
        d_scores[y[i]] -= 1
        for j in range(num_classes):
            d_score = d_scores[j]
            dW[:, j] += X[i].T * d_score

    loss /= num_train
    dW /= num_train

    loss += reg * np.sum(W * W)
    dW += reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    scores = X @ W  # shape (N, C)
    prob = np.exp(scores - np.max(scores, axis=1, keepdims=True)) / np.sum(np.exp(scores - np.max(scores, axis=1, keepdims=True)), axis=1, keepdims=True)
    correct_class_prob = prob[range(num_train), y]
    losses = - np.log(correct_class_prob)
    loss = np.sum(losses) / num_train

    loss += reg * np.sum(W * W)

    d_scores = prob
    d_scores[range(num_train), y] -= 1
    d_scores *= 1 / num_train # d_L_i = 1 / num_train
    dW += X.T @ d_scores

    dW += reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
