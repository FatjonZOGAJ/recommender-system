import numpy as np
import pandas as pd
from lib.models.base_model import BaseModel
import myfm
from easydict import EasyDict as edict
from myfm import RelationBlock
from scipy import sparse

from lib.utils.config import config

params = edict()
params.RANK = 32
params.N_ITER = 512  # does not work if not enough iterations
params.GROUPING = True
params.USE_IU = True  # use implicit user feature
params.USE_II = True  # use implicit item feature


class FMRelational(BaseModel):
    def fit(self, train_movies, train_users, train_predictions, **kwargs):
        self.df_train = pd.DataFrame({'user_id': train_users, 'movie_id': train_movies, 'rating': train_predictions})
        self.unique_user_ids = np.unique(self.df_train.user_id)
        self.unique_movie_ids = np.unique(self.df_train.movie_id)
        self.user_id_to_index = {uid: i for i, uid in enumerate(self.unique_user_ids)}
        self.movie_id_to_index = {mid: i for i, mid in enumerate(self.unique_movie_ids)}

        self.movie_vs_watched = dict()
        self.user_vs_watched = dict()
        for row in self.df_train.itertuples():
            user_id = row.user_id
            movie_id = row.movie_id
            self.movie_vs_watched.setdefault(movie_id, list()).append(user_id)
            self.user_vs_watched.setdefault(user_id, list()).append(movie_id)

        # setup grouping
        feature_group_sizes = []
        feature_group_sizes.append(len(self.user_id_to_index))  # user ids
        if params.USE_IU:
            feature_group_sizes.append(len(self.movie_id_to_index))
        feature_group_sizes.append(len(self.movie_id_to_index))  # movie ids
        if params.USE_II:
            feature_group_sizes.append(len(self.user_id_to_index))  # all users who watched the movies

        # Create RelationBlock.
        train_blocks = self._create_relational_blocks(self.df_train)

        self.fm = myfm.MyFMRegressor(rank=params.RANK)
        self.fm.fit(
            None, self.df_train.rating.values, X_rel=train_blocks,
            group_shapes=feature_group_sizes,
            n_iter=params.N_ITER
        )

    def predict(self, test_movies, test_users, save_submission):
        self.df_test = pd.DataFrame(
            {'user_id': test_users, 'movie_id': test_movies, 'ratings': np.zeros([len(test_users)], dtype=int)})

        # Create RelationBlock.
        test_blocks = self._create_relational_blocks(self.df_test)

        predictions = self.fm.predict(None, test_blocks)
        predictions[predictions >= 5] = 5
        predictions[predictions <= 1] = 1

        if save_submission:
            index = [''] * len(test_users)
            for i, (user, movie) in enumerate(zip(test_users, test_movies)):
                index[i] = f"r{user + 1}_c{movie + 1}"
            submission = pd.DataFrame({'Id': index, 'Prediction': predictions})
            submission.to_csv(config.SUBMISSION_NAME, index=False)

        return predictions

    def _create_relational_blocks(self, df):
        blocks = []
        for source, target in [(df, blocks)]:
            unique_users, user_map = np.unique(source.user_id, return_inverse=True)
            target.append(
                RelationBlock(user_map,
                              self._augment_user_id(unique_users, self.user_id_to_index, self.movie_id_to_index,
                                                    self.user_vs_watched))
            )
            unique_movies, movie_map = np.unique(source.movie_id, return_inverse=True)
            target.append(
                RelationBlock(movie_map,
                              self._augment_movie_id(unique_movies, self.movie_id_to_index, self.user_id_to_index,
                                                     self.movie_vs_watched))
            )

        return blocks

    def _augment_user_id(self, user_ids, user_id_to_index, movie_id_to_index, user_vs_watched):
        X = sparse.lil_matrix((len(user_ids), len(user_id_to_index) + (len(movie_id_to_index) if params.USE_IU else 0)))
        for index, user_id in enumerate(user_ids):
            if user_id in user_id_to_index:
                X[index, user_id_to_index[user_id]] = 1
            if not params.USE_IU:
                continue
            watched_movies = user_vs_watched.get(user_id, [])
            normalizer = 1 / max(len(watched_movies), 1) ** 0.5
            for mid in watched_movies:
                if mid in movie_id_to_index:
                    X[index, movie_id_to_index[mid] + len(user_id_to_index)] = normalizer
        return X.tocsr()

    def _augment_movie_id(self, movie_ids, movie_id_to_index, user_id_to_index, movie_vs_watched):
        X = sparse.lil_matrix(
            (len(movie_ids), len(movie_id_to_index) + (len(user_id_to_index) if params.USE_II else 0)))
        for index, movie_id in enumerate(movie_ids):
            if movie_id in movie_id_to_index:
                X[index, movie_id_to_index[movie_id]] = 1
            if not params.USE_II:
                continue
            watched_users = movie_vs_watched.get(movie_id, [])
            normalizer = 1 / max(len(watched_users), 1) ** 0.5
            for uid in watched_users:
                if uid in user_id_to_index:
                    X[index, user_id_to_index[uid] + len(movie_id_to_index)] = normalizer
        return X.tocsr()


def get_model(config, logger):
    return FMRelational()
