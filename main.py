import lib.models as models
from lib.utils import utils
from lib.utils.config import config
import pandas as pd
import numpy as np
import math
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def read_data():
    data_directory = config.DATA_DIR
    data_pd = pd.read_csv(f'{data_directory}/data_train.csv')
    print(data_pd.head(5))
    print()
    print('Shape', data_pd.shape)

    train_pd, test_pd = train_test_split(data_pd, train_size=config.TRAIN_SIZE, random_state=config.RANDOM_STATE)
    return train_pd, test_pd


def extract_users_items_predictions(data_pd):
    users, movies = \
        [np.squeeze(arr) for arr in
         np.split(data_pd.Id.str.extract('r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]
    predictions = data_pd.Prediction.values
    return users, movies, predictions


rmse = lambda x, y: math.sqrt(mean_squared_error(x, y))


# test our predictions with the true values
def get_score(predictions, target_values):  # =test_predictions):
    return rmse(predictions, target_values)


def extract_prediction_from_full_matrix(reconstructed_matrix,
                                        users, movies):
    # returns predictions for the users-movies combinations specified based on a full m \times n matrix
    assert (len(users) == len(movies)), "users-movies combinations specified should have equal length"
    predictions = np.zeros(len(users))

    for i, (user, movie) in enumerate(zip(users, movies)):
        predictions[i] = reconstructed_matrix[user][movie]

    return predictions


def main():
    logger = utils.init(seed=config.RANDOM_STATE)

    train_pd, test_pd = read_data()
    train_users, train_movies, train_predictions = extract_users_items_predictions(train_pd)
    test_users, test_movies, test_predictions = extract_users_items_predictions(test_pd)

    # create full matrix of observed values
    data, mask = create_matrices(train_movies, train_pd, train_predictions, train_users)

    logger.info(f'Using {config.MODEL} model for prediction')
    model = models.models[config.MODEL].get_model(config)

    predictions = model.predict(data, test_movies, test_users)
    print("RMSE using SVD is: {:.4f}".format(get_score(predictions, target_values=test_predictions)))


# TODO: use something else than train mean for default value
def create_matrices(train_movies, train_pd, train_predictions, train_users, default_replace='mean'):
    if default_replace == 'mean':
        default_val = np.mean(train_pd.Prediction.values)
    else:
        raise NotImplementedError('Add other replacement methods')
    data = np.full((config.NUM_USERS, config.NUM_MOVIES),
                   default_val)
    mask = np.zeros((config.NUM_USERS, config.NUM_MOVIES))  # 0 -> unobserved value, 1->observed value
    for user, movie, pred in zip(train_users, train_movies, train_predictions):
        data[user - 1][movie - 1] = pred
        mask[user - 1][movie - 1] = 1
    return data, mask


if __name__ == '__main__':
    main()
