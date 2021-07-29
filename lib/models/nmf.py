from sklearn.decomposition import non_negative_factorization

from lib.models.base_model import BaseModel
from lib.utils.config import config
from easydict import EasyDict as edict

params = edict()
params.RANK = None
params.INIT_PROCEDURE = 'nndsvda'
params.MAX_ITER = 100


class NMF(BaseModel):
    def __init__(self, config, logger, model_nr=0, rank=params.RANK):
        super().__init__(logger, model_nr)
        self.config = config
        params.RANK = rank

    def set_params(self, rank):
        params.RANK = rank
        return self

    def get_params(self, deep):
        return {"rank": params.RANK}

    def fit(self, train_movies, train_users, train_predictions, **kwargs):
        # create full matrix of observed and unobserved values
        data, mask = self.create_matrices(train_movies, train_users, train_predictions,
                                          default_replace=config.DEFAULT_VALUES[self.model_nr])
        W, H, _ = non_negative_factorization(data, n_components=params.RANK, init=params.INIT_PROCEDURE, verbose=True,
                                             max_iter=params.MAX_ITER)
        self.reconstructed_matrix = W @ H

    def predict(self, test_movies, test_users, save_submission, suffix='', postprocessing='default'):
        assert (len(test_users) == len(test_movies)), "users-movies combinations specified should have equal length"
        predictions, index = self._extract_prediction_from_full_matrix(self.reconstructed_matrix, users=test_users,
                                                                       movies=test_movies)
        predictions = self.postprocessing(predictions, postprocessing)
        if save_submission:
            self.save_submission(index, predictions, suffix)
        return predictions


def get_model(config, logger, model_nr=0):
    return NMF(config, logger, model_nr)
