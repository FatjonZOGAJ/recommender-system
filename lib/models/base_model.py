from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from lib import models
from lib.utils.config import config
from lib.utils.utils import roundPartial


class BaseModel(ABC):
    def __init__(self, logger, model_nr=0):
        self.logger = logger
        self.model_nr = model_nr

    @abstractmethod
    def fit(self, X, y, **kwargs):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def get_kwargs_data(self, kwargs, *keys):
        output = []
        for k in keys:
            output.append(kwargs[k] if k in kwargs else None)

        return output

    # necessary when instantiating model with no logger e.g. for initializing unobserved values
    def log_info(self, msg):
        if self.logger != None:
            self.logger.info(msg)
        else:
            print(msg)

    def _extract_prediction_from_full_matrix(self, reconstructed_matrix, users, movies):
        # returns predictions for the users-movies combinations specified based on a full m \times n matrix
        predictions = np.zeros(len(users))
        index = [''] * len(users)

        for i, (user, movie) in enumerate(zip(users, movies)):
            predictions[i] = reconstructed_matrix[user][movie]
            index[i] = f"r{user + 1}_c{movie + 1}"

        return predictions, index

    def save_submission(self, index, predictions, suffix=''):
        submission = pd.DataFrame({'Id': index, 'Prediction': predictions})
        filename = config.SUBMISSION_NAME
        if suffix != '':
            filename = config.SUBMISSION_NAME[:-4] + suffix + '.csv'
        submission.to_csv(filename, index=False)

    def create_matrices(self, train_movies, train_users, train_predictions, default_replace='mean'):
        data = np.full((config.NUM_USERS, config.NUM_MOVIES), 0, dtype=float)
        mask = np.zeros((config.NUM_USERS, config.NUM_MOVIES))  # 0 -> unobserved value, 1->observed value
        for user, movie, pred in zip(train_users, train_movies, train_predictions):
            data[user][movie] = pred
            mask[user][movie] = 1

        if default_replace == 'zero':
            pass
        elif default_replace == 'mean':
            data[mask == 0] = np.mean(train_predictions)
        elif default_replace == 'user_mean':
            for i in range(0, config.NUM_USERS):
                data[i, mask[i, :] == 0] = self.get_non_nan_mean(np.mean(data[i, :][mask[i, :] == 1]),
                                                                 np.mean(train_predictions))
        elif default_replace == 'item_mean':
            for i in range(0, config.NUM_MOVIES):
                data[mask[:, i] == 0, i] = self.get_non_nan_mean(np.mean(data[:, i][mask[:, i] == 1]),
                                                                 np.mean(train_predictions))
        else:
            if default_replace not in models.models:
                raise NotImplementedError('Add other replacement methods')

            unobserved_initializer_model = models.models[default_replace].get_model(config, logger=self.logger,
                                                                                    model_nr=self.model_nr + 1)
            data = self.use_model_to_init_unobserved(data, mask,
                                                     unobserved_initializer_model,
                                                     train_movies, train_predictions, train_users)
        self.log_info(f'Used {default_replace} to initialize unobserved entries as model {self.model_nr}')

        return data, mask

    def get_non_nan_mean(self, mean, default):
        # if no value for a movie/user (e.g. in validation split) TODO: more sophisticated
        if np.isnan(mean):
            self.log_info('RuntimeWarning due to no entries in masked row/col. Using overall mean.')
            mean = default
        return mean

    def use_model_to_init_unobserved(self, data, mask, unobserved_initializer_model, train_movies, train_predictions,
                                     train_users):
        unobserved_initializer_model.fit(train_movies, train_users, train_predictions)
        unobserved_indices = np.argwhere(mask == 0)
        unobserved_users, unobserved_movies = [unobserved_indices[:, c] for c in [0, 1]]
        predictions = unobserved_initializer_model.predict(unobserved_movies, unobserved_users, False,
                                                           postprocessing='default')
        for i in range(len(unobserved_indices)):
            user, movie = unobserved_indices[i]
            data[user][movie] = predictions[i]

        return data

    def postprocessing(self, predictions, type=''):
        if type == 'round_quarters':
            predictions = roundPartial(predictions, 0.25)
        elif type == 'nothing':
            self.log_info('Not doing postprocessing (no clipping either)')
            return predictions
        else:
            print('Postprocessing step not defined, only doing clipping', end='. ')

        predictions = np.clip(predictions, 1., 5.)
        print(f'Used {type} for postprocessing')
        return predictions
