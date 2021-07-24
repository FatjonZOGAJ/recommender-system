import pandas as pd

from lib import models
from lib.utils.config import config

SUBMISSION_CSV = '../../output/svd/2021-07-24-15-42-40/submission_2021-07-24-15-42-40.csv'


def postprocess(model, predictions, postprocessing, filename):
    postprocessed_pred = model.postprocessing(predictions.Prediction, postprocessing)
    config.SUBMISSION_NAME = filename
    model.save_submission(predictions.Id, postprocessed_pred, postprocessing)
    return postprocessed_pred


def postprocess_all(model, predictions, filename):
    for postprocessing in ['default', 'round_quarters', 'nothing']:
        postprocess(model, predictions, postprocessing, filename)


if __name__ == '__main__':
    model = models.models['svd'].get_model(config, None) # random model
    predictions = pd.read_csv(SUBMISSION_CSV)
    postprocess_all(model, predictions, filename=SUBMISSION_CSV)
