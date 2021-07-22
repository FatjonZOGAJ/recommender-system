import os
import numpy as np
import lib.models as models
from lib.utils import utils
from lib.utils.config import config
from lib.utils.loader import read_data, extract_users_items_predictions
from movie_distances import get_embeddings
from sklearn.cluster import KMeans

def perform_kmeans(embeddings):
    for num_cluster in range(10, 30):


if __name__ == '__main__':

    if os.path.exists('embeddings.npy'):
        embeddings = np.load('embeddings.npy')
    else:
        logger = utils.init(seed=config.RANDOM_STATE)
        logger.info(f'Using {config.MODEL} model for prediction')
        # Load data
        config.VALIDATE = False
        data_pd, test_pd = read_data()
        users, movies, predictions = extract_users_items_predictions(data_pd)
        model = models.models[config.MODEL].get_model(config, logger)
        data, mask = model.create_matrices(movies, users, predictions, config.DEFAULT_VALUE)

        embeddings = get_embeddings(data)
        np.save('movie_embeddings.npy', embeddings)

