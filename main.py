import lib.models as models
from lib.utils import utils
from lib.utils.utils import get_score

from lib.utils.config import config
from lib.utils.loader import read_data, extract_users_items_predictions, create_matrices


def main():
    logger = utils.init(seed=config.RANDOM_STATE)

    train_pd, val_pd, test_pd = read_data()
    train_users, train_movies, train_predictions = extract_users_items_predictions(train_pd)
    val_users, val_movies, val_predictions = extract_users_items_predictions(val_pd)
    test_users, test_movies, test_prediction = extract_users_items_predictions(test_pd)

    logger.info(f'Using {config.MODEL} model for prediction')
    model = models.models[config.MODEL].get_model(config)
    model.fit(train_movies, train_users, train_predictions)
    predictions = model.predict(val_movies, val_users, save_submission=False)
    logger.info("RMSE using SVD is: {:.4f}".format(get_score(predictions, target_values=val_predictions)))
    logger.info("Creating submission file")
    model.predict(test_movies, test_users, save_submission=True)


if __name__ == '__main__':
    main()
