import lib.models as models
from lib.utils import utils
from lib.utils.utils import get_score

from lib.utils.config import config
from lib.utils.loader import read_data, extract_users_items_predictions, create_matrices


def main():
    logger = utils.init(seed=config.RANDOM_STATE)

    train_pd, test_pd = read_data()
    train_users, train_movies, train_predictions = extract_users_items_predictions(train_pd)
    test_users, test_movies, test_predictions = extract_users_items_predictions(test_pd)

    # create full matrix of observed and unobserved values
    data, mask = create_matrices(train_movies, train_pd, train_predictions, train_users, default_replace=config.DEFAULT_VALUE)

    logger.info(f'Using {config.MODEL} model for prediction')
    model = models.models[config.MODEL].get_model(config)
    model.fit(data)
    predictions = model.predict(test_movies, test_users)
    print("RMSE using SVD is: {:.4f}".format(get_score(predictions, target_values=test_predictions)))


if __name__ == '__main__':
    main()
