from sklearn.decomposition import non_negative_factorization

from lib.models.base_model import BaseModel
from lib.utils.config import config


class ALS(BaseModel):
    def __init__(self, number_of_users, number_of_movies, logger, model_nr):
        super().__init__(logger, model_nr)
        self.num_users = number_of_users
        self.num_movies = number_of_movies

    def fit(self, train_movies, train_users, train_predictions, **kwargs):
        # create full matrix of observed and unobserved values
        data, mask = self.create_matrices(train_movies, train_users, train_predictions,
                                          default_replace=config.DEFAULT_VALUES[self.model_nr])
        W, H, _ = non_negative_factorization(data, verbose=True, max_iter=config.MAX_ITER)
        self.reconstructed_matrix = W @ H

    def predict(self, test_movies, test_users, save_submission):
        assert (len(test_users) == len(test_movies)), "users-movies combinations specified should have equal length"
        return self._extract_prediction_from_full_matrix(self.reconstructed_matrix, users=test_users,
                                                         movies=test_movies, save_submission=save_submission)


def get_model(config, logger, model_nr=0):
    return ALS(config.NUM_USERS, config.NUM_MOVIES, logger, model_nr)
