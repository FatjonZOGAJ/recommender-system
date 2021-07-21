import lib.models as models
from lib.utils import utils
from lib.utils.config import config
from lib.utils.loader import extract_users_items_predictions, read_data
from lib.utils.utils import get_score


def main():
    logger = utils.init(seed=config.RANDOM_STATE)
    logger.info(f'Using {config.MODEL} model for prediction')

    if config.VALIDATE:
        logger.info('Training on {:.0f}% of the data'.format(config.TRAIN_SIZE * 100))
        train_pd, val_pd, test_pd = read_data()
        train_users, train_movies, train_predictions = extract_users_items_predictions(train_pd)
        val_users, val_movies, val_predictions = extract_users_items_predictions(val_pd)
        test_users, test_movies, _ = extract_users_items_predictions(test_pd)
        model = models.models[config.MODEL].get_model(config, logger)
        logger.info("Fitting the model")
        model.fit(train_movies, train_users, train_predictions,
                  val_movies=val_movies, val_users=val_users, val_predictions=val_predictions)  # iterative val score
        logger.info("Testing the model")
        predictions = model.predict(val_movies, val_users, save_submission=False)
        logger.info('RMSE using {} is {:.4f}'.format(
            config.MODEL, get_score(predictions, target_values=val_predictions)))
    else:
        logger.info('Training on 100% of the data')
        train_pd, test_pd = read_data()
        train_users, train_movies, train_predictions = extract_users_items_predictions(train_pd)
        test_users, test_movies, _ = extract_users_items_predictions(test_pd)
        model = models.models[config.MODEL].get_model(config, logger)
        logger.info("Fitting the model")
        model.fit(train_movies, train_users, train_predictions,
                  test_movies=test_movies, test_users=test_users, test_every=config.TEST_EVERY)  # iterative test score

    logger.info("Creating submission file")
    model.predict(test_movies, test_users, save_submission=True)


if __name__ == '__main__':
    main()
