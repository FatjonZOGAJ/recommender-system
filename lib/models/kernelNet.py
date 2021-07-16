'''
adapted from Lorenz Muller
https://github.com/lorenzMuller/kernelNet_MovieLens/blob/master/kernelNet_ml1m.py
'''
import numpy as np
import pandas as pd
import tensorflow as tf
from time import time
import sys
# from dataLoader import loadData
import os

from easydict import EasyDict as edict

from lib.models.base_model import BaseModel
from lib.utils.config import config

# TODO: add to params
params = edict()
params.HIDDEN_UNITS = 500
LAMBDA_2 = 60.  # float(sys.argv[1]) if len(sys.argv) > 1 else 60.
LAMBDA_SPARSITY = 0.013  # float(sys.argv[2]) if len(sys.argv) > 2 else 0.013
N_LAYERS = 2
OUTPUT_EVERY = 50  # evaluate performance on test set; breaks l-bfgs loop
N_EPOCHS = N_LAYERS * 10 * OUTPUT_EVERY
VERBOSE_BFGS = True


# use_gpu = True
# if not use_gpu:
#     os.environ['CUDA_VISIBLE_DEVICES'] = ''


class KernelNet(BaseModel):
    def __init__(self):
        self.explanation_columns = ['user_id', 'movie_id']

        # Input placeholders
        self.R = tf.placeholder("float", [None, config.NUM_USERS])

        # Instantiate network
        y = self.R
        reg_losses = None
        for i in range(N_LAYERS):
            y, reg_loss = self.kernel_layer(y, params.HIDDEN_UNITS, name=str(i), lambda_2=LAMBDA_2,
                                            lambda_s=LAMBDA_SPARSITY)
            reg_losses = reg_loss if reg_losses is None else reg_losses + reg_loss
        self.prediction, reg_loss = self.kernel_layer(y, config.NUM_USERS, activation=tf.identity, name='out',
                                                      lambda_2=LAMBDA_2, lambda_s=LAMBDA_SPARSITY)
        self.reg_losses = reg_losses + reg_loss

    def fit_init(self, train_mask):
        # Compute loss (symbolic)
        diff = train_mask * (self.R - self.prediction)
        sqE = tf.nn.l2_loss(diff)
        loss = sqE + self.reg_losses

        # Instantiate L-BFGS Optimizer
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss, options={'maxiter': OUTPUT_EVERY,
                                                                               'disp': VERBOSE_BFGS,
                                                                               'maxcor': 10},
                                                                method='L-BFGS-B')

    # define network functions
    def kernel(self, u, v):
        """
        Sparsifying kernel function

        :param u: input vectors [n_in, 1, n_dim]
        :param v: output vectors [1, hidden_units, n_dim]
        :return: input to output connection matrix
        """
        dist = tf.norm(u - v, ord=2, axis=2)
        hat = tf.maximum(0., 1. - dist ** 2)
        return hat

    def kernel_layer(self, x, hidden_units=500, n_dim=5, activation=tf.nn.sigmoid, lambda_s=0.013,
                     lambda_2=60., name=''):
        """
        a kernel sparsified layer

        :param x: input [batch, channels]
        :param hidden_units: number of hidden units
        :param n_dim: number of dimensions to embed for kernelization
        :param activation: output activation
        :param name: layer name for scoping
        :return: layer output, regularization term
        """

        # define variables
        with tf.variable_scope(name):
            W = tf.get_variable('W', [x.shape[1], hidden_units])
            n_in = x.get_shape().as_list()[1]
            u = tf.get_variable('u', initializer=tf.random.truncated_normal([n_in, 1, n_dim], 0., 1e-3))
            v = tf.get_variable('v', initializer=tf.random.truncated_normal([1, hidden_units, n_dim], 0., 1e-3))
            b = tf.get_variable('b', [hidden_units])

        # compute sparsifying kernel
        # as u and v move further from each other for some given pair of neurons, their connection
        # decreases in strength and eventually goes to zero.
        w_hat = self.kernel(u, v)

        # compute regularization terms
        sparse_reg = tf.contrib.layers.l2_regularizer(lambda_s)
        sparse_reg_term = tf.contrib.layers.apply_regularization(sparse_reg, [w_hat])

        l2_reg = tf.contrib.layers.l2_regularizer(lambda_2)
        l2_reg_term = tf.contrib.layers.apply_regularization(l2_reg, [W])

        # compute output
        W_eff = W * w_hat
        y = tf.matmul(x, W_eff) + b
        y = activation(y)
        return y, sparse_reg_term + l2_reg_term

    # data must be
    def fit(self, train_movies, train_users, train_predictions):
        # NOTE transposed
        data_train = np.full((config.NUM_USERS, config.NUM_MOVIES), 0, dtype=float).transpose()
        mask_train = np.zeros((config.NUM_USERS, config.NUM_MOVIES)).transpose()
        for user, movie, pred in zip(train_users, train_movies, train_predictions):
            data_train[movie][user] = pred
            mask_train[movie][user] = 1
        # data_train = pd.DataFrame({'user_id': train_users, 'movie_id': train_movies, 'rating': train_predictions})
        # mask_train = data_train.mask(data_train != 0, 1).to_numpy()
        # data_train = data_train.to_numpy()

        #  TOOD maybe to ndarray
        self.fit_init(mask_train)

        # Training and validation loop
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for i in range(int(N_EPOCHS / OUTPUT_EVERY)):
                self.optimizer.minimize(sess, feed_dict={self.R: data_train})  # do maxiter optimization steps
                pre = sess.run(self.prediction, feed_dict={self.R: data_train})  # predict ratings

                error = 0  # (vm * (np.clip(pre, 1., 5.) - vr) ** 2).sum() / vm.sum()  # compute validation error
                error_train = (mask_train * (
                        np.clip(pre, 1., 5.) - data_train) ** 2).sum() / mask_train.sum()  # compute train error

                print('.-^-._' * 12)
                print('epoch:', i, 'validation rmse:', np.sqrt(error), 'train rmse:', np.sqrt(error_train))
                print('.-^-._' * 12)

            with open('summary_ml1m.txt', 'a') as file:
                for a in sys.argv[1:]:
                    file.write(a + ' ')
                file.write(str(np.sqrt(error)) + ' ' + str(np.sqrt(error_train))
                           + ' ' + str(config.RANDOM_STATE) + '\n')
                file.close()

    def predict(self, test_movies, test_users, save_submission):
        ...
        df_train = pd.DataFrame({'user_id': test_users, 'movie_id': test_movies})

        # if save_submission:
        #     index = [''] * len(test_users)
        #     for i, (user, movie) in enumerate(zip(test_users, test_movies)):
        #         index[i] = f"r{user + 1}_c{movie + 1}"
        #     submission = pd.DataFrame({'Id': index, 'Prediction': predictions})
        #     submission.to_csv(config.SUBMISSION_NAME, index=False)
        # return predictions


def get_model(config):
    return KernelNet()
