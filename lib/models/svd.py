import numpy as np

from lib.models.base_model import BaseModel
from lib.utils.config import config
from easydict import EasyDict as edict

params = edict()
params.RANK = 3


class SVD(BaseModel):
    def __init__(self, config, logger, model_nr=0, rank=params.RANK):
        super().__init__(logger, model_nr)
        self.num_users = config.NUM_USERS
        self.num_movies = config.NUM_MOVIES
        params.RANK = rank

    def fit(self, train_movies, train_users, train_predictions, **kwargs):
        # create full matrix of observed and unobserved values
        data, mask = self.create_matrices(train_movies, train_users, train_predictions,
                                          default_replace=config.DEFAULT_VALUES[self.model_nr])
        number_of_singular_values = min(self.num_users, self.num_movies)
        assert (params.RANK <= number_of_singular_values), "choose correct number of singular values"
        U, s, Vt = np.linalg.svd(data, full_matrices=False)
        S = np.zeros((self.num_movies, self.num_movies))
        S[:params.RANK, :params.RANK] = np.diag(s[:params.RANK])
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
    return SVD(config, logger, model_nr)
