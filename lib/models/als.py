import pandas as pd
import numpy as np

from lib.models.base_model import BaseModel
from sklearn.decomposition import non_negative_factorization

from lib.utils.config import config
from lib.utils.loader import create_matrices


class ALS(BaseModel):
    def __init__(self, number_of_users, number_of_movies):
        self.num_users = number_of_users
        self.num_movies = number_of_movies

    def fit(self, train_movies, train_users, train_predictions):
        # create full matrix of observed and unobserved values
        data, mask = create_matrices(train_movies, train_users, train_predictions,
                                     default_replace=config.DEFAULT_VALUE)
        W, H, _ = non_negative_factorization(data, verbose=True)
        self.reconstructed_matrix = W @ H

    def predict(self, test_movies, test_users, save_submission):
        assert (len(test_users) == len(test_movies)), "users-movies combinations specified should have equal length"
        return self._extract_prediction_from_full_matrix(self.reconstructed_matrix, users=test_users,
                                                         movies=test_movies, save_submission=save_submission)

    def _extract_prediction_from_full_matrix(self, reconstructed_matrix, users, movies, save_submission=True):
        # returns predictions for the users-movies combinations specified based on a full m \times n matrix
        predictions = np.zeros(len(users))
        index = [''] * len(users)

        for i, (user, movie) in enumerate(zip(users, movies)):
            predictions[i] = reconstructed_matrix[user][movie]
            index[i] = f"r{user + 1}_c{movie + 1}"

        if save_submission:
            submission = pd.DataFrame({'Id': index, 'Prediction': predictions})
            submission.to_csv(config.SUBMISSION_NAME, index=False)
        return predictions


def get_model(config):
    return ALS(config.NUM_USERS, config.NUM_MOVIES)