import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from lib.utils.config import config


def read_data():
    data_directory = config.DATA_DIR
    data_pd = pd.read_csv(f'{data_directory}/data_train.csv')
    print(data_pd.head(5))
    print()
    print('Shape', data_pd.shape)

    test_pd = pd.read_csv(f'{data_directory}/sampleSubmission.csv')

    if config.VALIDATE:
        train_pd, val_pd = train_test_split(data_pd, train_size=config.TRAIN_SIZE, random_state=config.RANDOM_STATE)
        return train_pd, val_pd, test_pd
    else:
        return data_pd, test_pd


def extract_users_items_predictions(data_pd):
    users, movies = \
        [np.squeeze(arr) for arr in
         np.split(data_pd.Id.str.extract('r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]
    predictions = data_pd.Prediction.values
    # index for users and movies starts at 0
    return users, movies, predictions


def create_matrices(train_movies, train_users, train_predictions, default_replace='mean'):
    data = np.full((config.NUM_USERS, config.NUM_MOVIES), 0, dtype=float)
    mask = np.zeros((config.NUM_USERS, config.NUM_MOVIES))  # 0 -> unobserved value, 1->observed value
    for user, movie, pred in zip(train_users, train_movies, train_predictions):
        data[user][movie] = pred
        mask[user][movie] = 1

    if default_replace == 'mean':
        data[mask == 0] = np.mean(train_predictions)
    elif default_replace == 'user_mean':
        for i in range(0, config.NUM_USERS):
            mean_of_i_row = np.mean(data[i, :][mask[i, :] == 1])
            data[i, mask[i, :] == 0] = mean_of_i_row
    elif default_replace == 'item_mean':
        for i in range(0, config.NUM_MOVIES):
            mean_of_i_col = np.mean(data[:, i][mask[:, i] == 1])
            data[mask[:, i] == 0, i] = mean_of_i_col
    else:
        raise NotImplementedError('Add other replacement methods')
    return data, mask