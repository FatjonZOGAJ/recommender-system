import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from lib.models.base_model import BaseModel
import myfm
from easydict import EasyDict as edict

from lib.utils.config import config

params = edict()
params.RANK = 8
params.N_ITER = 200
params.SAMPLES = 195
params.GROUPING = True


class FM(BaseModel):
    def __init__(self):
        self.explanation_columns = ['user_id', 'movie_id']
        self.fm = myfm.MyFMRegressor(rank=params.RANK, random_seed=69)

    def fit(self, train_movies, train_users, train_predictions, **kwargs):
        df_train = pd.DataFrame({'user_id': train_users, 'movie_id': train_movies, 'rating': train_predictions})
        self.ohe = OneHotEncoder(handle_unknown="ignore")
        X_train = self.ohe.fit_transform(df_train[self.explanation_columns])
        y_train = df_train.rating.values

        if params.GROUPING:
            # specify how columns of X_train are grouped
            group_shapes = [len(category) for category in self.ohe.categories_]
            assert sum(group_shapes) == X_train.shape[1]
        else:
            group_shapes = None

        self.fm.fit(
            X_train,
            y_train,
            group_shapes=group_shapes,
            n_iter=params.N_ITER,
            n_kept_samples=params.SAMPLES,
        )

    def predict(self, test_movies, test_users, save_submission):
        df_train = pd.DataFrame({'user_id': test_users, 'movie_id': test_movies})
        X_test = self.ohe.transform(df_train[self.explanation_columns])
        predictions = self.fm.predict(X_test)

        if save_submission:
            index = [''] * len(test_users)
            for i, (user, movie) in enumerate(zip(test_users, test_movies)):
                index[i] = f"r{user + 1}_c{movie + 1}"
            submission = pd.DataFrame({'Id': index, 'Prediction': predictions})
            submission.to_csv(config.SUBMISSION_NAME, index=False)
        return predictions


def get_model(config, logger):
    return FM()
