from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from lib.utils.config import config


class BaseModel(ABC):
    @abstractmethod
    def fit(self, train_movies, train_users, train_predictions, **kwargs):
        pass

    @abstractmethod
    def predict(self, test_movies, test_users, save_submission):
        pass

    def get_kwargs_data(self, kwargs, *keys):
        output = []
        for k in keys:
            output.append(kwargs[k] if k in kwargs else None)

        return output

    def _extract_prediction_from_full_matrix(self, reconstructed_matrix, users, movies, save_submission=True,
                                             suffix=''):
        # returns predictions for the users-movies combinations specified based on a full m \times n matrix
        predictions = np.zeros(len(users))
        index = [''] * len(users)

        for i, (user, movie) in enumerate(zip(users, movies)):
            predictions[i] = reconstructed_matrix[movie][user]
            index[i] = f"r{user + 1}_c{movie + 1}"

        if save_submission:
            submission = pd.DataFrame({'Id': index, 'Prediction': predictions})
            filename = config.SUBMISSION_NAME
            if suffix != '':
                filename = config.SUBMISSION_NAME[:-4] + suffix + '.csv'
            submission.to_csv(filename, index=False)
        return predictions
