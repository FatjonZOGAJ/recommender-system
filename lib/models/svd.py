import numpy as np

from lib.models.base_model import BaseModel
from lib.utils.config import config


class SVD(BaseModel):
    def __init__(self, number_of_users, number_of_movies, logger, model_nr):
        super().__init__(logger, model_nr)
        self.num_users = number_of_users
        self.num_movies = number_of_movies

    def fit(self, train_movies, train_users, train_predictions, **kwargs):
        # create full matrix of observed and unobserved values
        data, mask = self.create_matrices(train_movies, train_users, train_predictions,
                                          default_replace=config.DEFAULT_VALUES[self.model_nr])
        k_singular_values = config.K_SINGULAR_VALUES
        number_of_singular_values = min(self.num_users, self.num_movies)
        assert (k_singular_values <= number_of_singular_values), "choose correct number of singular values"
        U, s, Vt = np.linalg.svd(data, full_matrices=False)
        S = np.zeros((self.num_movies, self.num_movies))
        S[:k_singular_values, :k_singular_values] = np.diag(s[:k_singular_values])
        self.reconstructed_matrix = U.dot(S).dot(Vt)

    def predict(self, test_movies, test_users, save_submission, suffix='', postprocessing='default'):
        assert (len(test_users) == len(test_movies)), "users-movies combinations specified should have equal length"
        predictions, index = self._extract_prediction_from_full_matrix(self.reconstructed_matrix, users=test_users,
                                                         movies=test_movies)
        predictions = self.postprocessing(predictions, postprocessing)
        if save_submission:
            self.save_submission(index, predictions, suffix)
        return predictions



def get_model(config, logger, model_nr=0):
    return SVD(config.NUM_USERS, config.NUM_MOVIES, logger, model_nr)
