'''
adapted from Lorenz Muller
https://github.com/lorenzMuller/kernelNet_MovieLens/blob/master/kernelNet_ml1m.py
'''
import numpy as np
import tensorflow as tf
# from dataLoader import loadData

from easydict import EasyDict as edict

from lib.models.base_model import BaseModel
from lib.utils.config import config

# TODO: add to params

TESTING = False

params = edict()
params.HIDDEN_UNITS = 500
LAMBDA_2 = 90.
LAMBDA_SPARSITY = 0.023
N_LAYERS = 2
OUTPUT_EVERY = 50 if not TESTING else 5  # evaluate performance on test set; breaks l-bfgs loop
N_EPOCHS = 3# N_LAYERS * 10
VERBOSE_BFGS = False


class KernelNet(BaseModel):
    def __init__(self, logger, is_initializer_model):
        super().__init__(logger, is_initializer_model)

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

    def fit(self, train_movies, train_users, train_predictions, **kwargs):
        val_movies, val_users, val_predictions = self.get_kwargs_data(kwargs, 'val_movies', 'val_users',
                                                                      'val_predictions')
        test_movies, test_users, test_every = self.get_kwargs_data(kwargs, 'test_movies', 'test_users', 'test_every')
        # TODO: zero better?
        data, mask = self.create_matrices(train_movies, train_users, train_predictions,
                                     default_replace=config.DEFAULT_VALUE if not self.is_initializer_model
                                     else config.SECOND_DEFAULT_VALUE)
        data, mask = data.T, mask.T  # NOTE transpose
        if not val_movies is None:
            data_val, mask_val = self.create_matrices(val_movies, val_users, val_predictions,
                                         default_replace=config.DEFAULT_VALUE if not self.is_initializer_model
                                         else config.SECOND_DEFAULT_VALUE)
            data_val, mask_val = data_val.T, mask_val.T  # NOTE transpose

        self.fit_init(mask)

        # Training and validation loop
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for i in range(N_EPOCHS):
                self.optimizer.minimize(sess, feed_dict={self.R: data})  # do maxiter optimization steps
                pre = sess.run(self.prediction, feed_dict={self.R: data})  # predict ratings

                error_val = 0 if val_movies is None \
                    else (mask_val * (np.clip(pre, 1., 5.) - data_val) ** 2).sum() / mask_val.sum()
                error_train = (mask * (np.clip(pre, 1., 5.) - data) ** 2).sum() / mask.sum()

                print('.-^-._' * 12)
                self.log_info(
                    f'epoch: {i}, validation rmse: {np.sqrt(error_val)} train rmse: {np.sqrt(error_train)}')

                self.reconstructed_matrix = pre

                if i == 0 and TESTING:
                    break

                if not test_movies is None and i + 1 % test_every == 0:
                    self.log_info(f'Creating submission for epoch {i} with train_err {np.sqrt(error_train)}')
                    self.predict(test_movies, test_users, True, suffix=f'_e{i}_err{np.sqrt(error_train):.2f}')

    def predict(self, test_movies, test_users, save_submission, suffix=''):
        assert (len(test_users) == len(test_movies)), "users-movies combinations specified should have equal length"
        return self._extract_prediction_from_full_matrix(self.reconstructed_matrix.transpose(), users=test_users,
                                                         movies=test_movies, save_submission=save_submission,
                                                         suffix=suffix)


def get_model(config, logger, is_initializer_model=False):
    return KernelNet(logger, is_initializer_model)
