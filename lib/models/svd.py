import numpy as np

from main import extract_prediction_from_full_matrix


class SVD:
    def __init__(self, number_of_users, number_of_movies):
        self.num_users = number_of_users
        self.num_movies = number_of_movies

    def predict(self, data, test_movies, test_users):
        k_singular_values = 2
        number_of_singular_values = min(self.num_users, self.num_movies)
        assert (k_singular_values <= number_of_singular_values), "choose correct number of singular values"
        U, s, Vt = np.linalg.svd(data, full_matrices=False)
        S = np.zeros((self.num_movies, self.num_movies))
        S[:k_singular_values, :k_singular_values] = np.diag(s[:k_singular_values])
        reconstructed_matrix = U.dot(S).dot(Vt)

        predictions = extract_prediction_from_full_matrix(reconstructed_matrix, users=test_users, movies=test_movies)
        return predictions


def get_model(config):
    return SVD(config.NUM_USERS, config.NUM_MOVIES)
