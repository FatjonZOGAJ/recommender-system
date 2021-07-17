from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def fit(self, train_movies, train_users, train_predictions, **kwargs):
        pass

    @abstractmethod
    def predict(self, test_movies, test_users, save_submission):
        pass
