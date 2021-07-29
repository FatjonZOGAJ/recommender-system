import os

import matplotlib.pyplot as plt
import numpy as np
from lib.models import models
from lib.utils import utils
from lib.utils.config import config
from lib.utils.loader import extract_users_items_predictions, read_data
from lib.utils.utils import get_score
import time


def plot_heatmap(matrix, x_values, y_values, show_values=False):
    fig, ax = plt.subplots()
    im = ax.imshow(matrix, interpolation='nearest', cmap='winter')
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(len(x_values)),
           yticks=np.arange(len(y_values)),
           xticklabels=x_values, yticklabels=y_values,
           title="Heatmap",
           ylabel='y_values',
           xlabel='x_values')

    if show_values:
        fmt = '.2f'
        thresh = matrix.max() / 2.
        for i in range(len(x_values)):
            for j in range(len(y_values)):
                ax.text(j, i, format(matrix[i, j], fmt),
                        ha="center", va="center",
                        color="white" if matrix[i, j] > thresh else "black")

    fig.savefig('heatmap.png')


def plot_autoencoder_rmse_single_layer(encoded_dimension, rmse, path):
    plt.plot(encoded_dimension, rmse, '-x')
    plt.title('Single hidden layer Autoencoder')

    plt.xlabel('Encoded dimension')
    plt.ylabel('RMSE')
    plt.savefig(path)


def call_autoencoder_rmse_single_layer(train_users, train_movies, train_predictions, val_users,
                                       val_movies, val_predictions):
    encoded_dimensions = [16, 32, 50, 100, 250, 350, 500]
    final_rmse = []

    for encoded_dimension in encoded_dimensions:
        config.ENCODED_DIMENSION = encoded_dimension
        model = models[config.MODEL].get_model(config, logger)
        model.fit(train_movies, train_users, train_predictions,
                  val_movies=val_movies, val_users=val_users, val_predictions=val_predictions)  # iterative val score
        predictions = model.predict(val_movies, val_users, save_submission=False, postprocessing='default')
        rmse = get_score(predictions, target_values=val_predictions)
        final_rmse.append(rmse)

    plot_autoencoder_rmse_single_layer(encoded_dimensions, final_rmse, 'autoencoder_single_layer.png')


def plot_deep_autoencoder(encoded_dimension, rmse, path):
    plt.plot(encoded_dimension, rmse[0], '-x', c='b', label='1 layer')
    plt.plot(encoded_dimension, rmse[1], '-D', c='r', label='2 layers')

    plt.xlabel('Encoded dimension')
    plt.ylabel('RMSE')
    plt.legend()
    plt.savefig(path)


def call_deep_autoencoder(train_users, train_movies, train_predictions, val_users, val_movies, val_predictions):
    encoded_dimensions = [16, 32, 50, 100, 250, 350, 500]
    one_layer_rmse = []
    two_layer_rmse = []

    config.SINGLE_LAYER = True
    for encoded_dimension in encoded_dimensions:
        config.ENCODED_DIMENSION = encoded_dimension
        model = models[config.MODEL].get_model(config, logger)
        model.fit(train_movies, train_users, train_predictions,
                  val_movies=val_movies, val_users=val_users, val_predictions=val_predictions)  # iterative val score
        predictions = model.predict(val_movies, val_users, save_submission=False, postprocessing='default')
        rmse = get_score(predictions, target_values=val_predictions)
        one_layer_rmse.append(rmse)

    config.SINGLE_LAYER = False
    config.HIDDEN_DIMENSION = [500]
    for encoded_dimension in encoded_dimensions:
        config.ENCODED_DIMENSION = encoded_dimension
        model = models[config.MODEL].get_model(config, logger)
        model.fit(train_movies, train_users, train_predictions,
                  val_movies=val_movies, val_users=val_users, val_predictions=val_predictions)  # iterative val score
        predictions = model.predict(val_movies, val_users, save_submission=False, postprocessing='default')
        rmse = get_score(predictions, target_values=val_predictions)
        two_layer_rmse.append(rmse)

    plot_deep_autoencoder(encoded_dimensions, [one_layer_rmse, two_layer_rmse], 'deep_autoencoder.png')


def plot_rmse_embedding(embedding_dimensions, rmse_values, markers, colors, labels, path):
    for i in range(len(rmse_values)):
        plt.plot(embedding_dimensions, rmse_values[i], markers[i], c=colors[i], label=labels[i])
    plt.legend()
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Test RMSE')
    plt.savefig(path)


def call_rmse_embedding(train_users, train_movies, train_predictions, val_users, val_movies, val_predictions):
    embedding_dimensions = [2, 4, 10, 16, 32, 64, 128]
    rmse_values = []
    markers = []
    colors = []
    labels = []

    config.MODEL = 'svd'

    svd_rmse = []
    for embedding_dimension in embedding_dimensions:
        config.K_SINGULAR_VALUES = embedding_dimension
        model = models[config.MODEL].get_model(config, logger)
        model.fit(train_movies, train_users, train_predictions,
              val_movies=val_movies, val_users=val_users, val_predictions=val_predictions)  # iterative val score
        predictions = model.predict(val_movies, val_users, save_submission=False, postprocessing='default')
        rmse = get_score(predictions, target_values=val_predictions)
        svd_rmse.append(rmse)
    rmse_values.append(svd_rmse)
    colors.append('red')
    markers.append('-D')
    labels.append('svd')

    # TODO: repeat for other algorithms

    plot_rmse_embedding(embedding_dimensions, rmse_values, markers, colors, labels, 'rmse_embedding.png')


def plot_rmse_validation(epochs, rmse_values, colors, labels, path):
    for i in range(len(rmse_values)):
        plt.plot(epochs, rmse_values[i], c=colors[i], label=labels[i])
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Validation RMSE')
    plt.savefig(path)


def call_rmse_validation(train_users, train_movies, train_predictions, val_users, val_movies, val_predictions):
    model_names = ['autoencoder', 'autorec', 'ncf', 'kernel_net']
    rmse_values = []
    times = []
    for model_name in model_names:
        if os.path.exists(model_name + '.npy'):
            data = np.load(model_name + '.npy')
            rmse_values.append(data.tolist())
        else:
            config.MODEL = model_name
            model = models[config.MODEL].get_model(config, logger)
            start = time.time()
            model.fit(train_movies, train_users, train_predictions,
                      val_movies=val_movies, val_users=val_users, val_predictions=val_predictions)
            end = time.time()
            times.append(end-start)
            np.save(model_name + '.npy', np.array(model.validation_rmse))
            rmse_values.append(model.validation_rmse)
    np.save('validation_rmse.npy', np.array(rmse_values))
    np.save('times.npy', times)
    colors = ['red', 'blue', 'green', 'orange']
    plot_rmse_validation(range(0, config.NUM_EPOCHS, config.TEST_EVERY), rmse_values, colors, model_names, 'validation_plot.ong')


if __name__ == '__main__':
    logger = utils.init(seed=config.RANDOM_STATE)
    logger.info(f'Using {config.MODEL} model for prediction')
    # Load data
    train_pd, val_pd, test_pd = read_data()
    train_users, train_movies, train_predictions = extract_users_items_predictions(train_pd)
    val_users, val_movies, val_predictions = extract_users_items_predictions(val_pd)
    test_users, test_movies, _ = extract_users_items_predictions(test_pd)

    call_rmse_validation(train_users, train_movies, train_predictions, val_users, val_movies, val_predictions)

    # call_rmse_embedding(train_users, train_movies, train_predictions, val_users, val_movies, val_predictions)
    # call_autoencoder_rmse_single_layer(train_users, train_movies, train_predictions, val_users, val_movies, val_predictions)
    #
    # plot_heatmap(np.random.rand(3,2), [1, 2], [4, 5, 6])