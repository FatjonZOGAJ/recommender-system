import numpy as np
import pandas as pd

from lib.utils.config import config


class SVD:
    def __init__(self, number_of_users, number_of_movies):
        self.num_users = number_of_users
        self.num_movies = number_of_movies

    def fit(self, data):
        k_singular_values = 2
        number_of_singular_values = min(self.num_users, self.num_movies)
        assert (k_singular_values <= number_of_singular_values), "choose correct number of singular values"
        U, s, Vt = np.linalg.svd(data, full_matrices=False)
        S = np.zeros((self.num_movies, self.num_movies))
        S[:k_singular_values, :k_singular_values] = np.diag(s[:k_singular_values])
        self.reconstructed_matrix = U.dot(S).dot(Vt)

    def predict(self, test_movies, test_users, save_submission=True):
        return self._extract_prediction_from_full_matrix(self.reconstructed_matrix, users=test_users,
                                                         movies=test_movies, save_submission=save_submission)

    def _extract_prediction_from_full_matrix(self, reconstructed_matrix, users, movies, save_submission=True):
        # returns predictions for the users-movies combinations specified based on a full m \times n matrix
        assert (len(users) == len(movies)), "users-movies combinations specified should have equal length"
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
    return SVD(config.NUM_USERS, config.NUM_MOVIES)
