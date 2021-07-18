import numpy as np
import pandas as pd

from lib.utils.config import config
from lib.models.base_model import BaseModel
from lib.utils.loader import create_matrices


class SVD(BaseModel):
    def __init__(self, number_of_users, number_of_movies):
        self.num_users = number_of_users
        self.num_movies = number_of_movies

    def fit(self, train_movies, train_users, train_predictions, **kwargs):
        # create full matrix of observed and unobserved values
        data, mask = create_matrices(train_movies, train_users, train_predictions,
                                     default_replace=config.DEFAULT_VALUE)
        k_singular_values = config.K_SINGULAR_VALUES
        number_of_singular_values = min(self.num_users, self.num_movies)
        assert (k_singular_values <= number_of_singular_values), "choose correct number of singular values"
        U, s, Vt = np.linalg.svd(data, full_matrices=False)
        S = np.zeros((self.num_movies, self.num_movies))
        S[:k_singular_values, :k_singular_values] = np.diag(s[:k_singular_values])
        self.reconstructed_matrix = U.dot(S).dot(Vt)

    def predict(self, test_movies, test_users, save_submission):
        assert (len(test_users) == len(test_movies)), "users-movies combinations specified should have equal length"
        return self._extract_prediction_from_full_matrix(self.reconstructed_matrix, users=test_users,
                                                         movies=test_movies, save_submission=save_submission)


def get_model(config, logger):
    return SVD(config.NUM_USERS, config.NUM_MOVIES)
