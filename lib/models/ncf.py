from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from lib.models.base_model import BaseModel
import torch
import torch.nn as nn
from easydict import EasyDict as edict

from lib.utils.utils import get_score

params = edict()
params.embedding_size = 16


class NCF(BaseModel):
    def __init__(self, config, logger, ncf_network, device):
        super().__init__(logger)
        self.config = config
        self.ncf_network = ncf_network
        self.device = device

    def fit(self, train_movies, train_users, train_predictions, **kwargs):
        train_users_torch = torch.tensor(train_users, device=self.device)
        train_movies_torch = torch.tensor(train_movies, device=self.device)
        train_predictions_torch = torch.tensor(train_predictions, device=self.device)

        train_dataloader = DataLoader(
            TensorDataset(train_users_torch, train_movies_torch, train_predictions_torch),
            batch_size=self.config.BATCH_SIZE)

        if self.config.VALIDATE:
            test_users_torch = torch.tensor(kwargs['val_users'], device=self.device)
            test_movies_torch = torch.tensor(kwargs['val_movies'], device=self.device)

            test_dataloader = DataLoader(
                TensorDataset(test_users_torch, test_movies_torch),
                batch_size=self.config.BATCH_SIZE)

        optimizer = optim.Adam(self.ncf_network.parameters(),
                               lr=self.config.LEARNING_RATE)

        step = 0
        for epoch in range(self.config.NUM_EPOCHS):
            for users_batch, movies_batch, target_predictions_batch in train_dataloader:
                optimizer.zero_grad()
                predictions_batch = self.ncf_network(users_batch, movies_batch)
                loss = mse_loss(predictions_batch, target_predictions_batch)
                loss.backward()
                optimizer.step()
                step += 1

            if epoch % self.config.TEST_EVERY == 0 and self.config.VALIDATE:
                with torch.no_grad():
                    all_predictions = []
                    for users_batch, movies_batch in test_dataloader:
                        predictions_batch = self.ncf_network(users_batch, movies_batch)
                        all_predictions.append(predictions_batch)

                all_predictions = torch.cat(all_predictions)

                reconstuction_rmse = get_score(all_predictions.cpu().numpy(), kwargs['val_predictions'])
                self.logger.info('At epoch {:3d} loss is {:.4f}'.format(epoch, reconstuction_rmse))
                self.validation_rmse.append(reconstuction_rmse)

    def predict(self, test_movies, test_users, save_submission, suffix='', postprocessing='default'):
        test_users_torch = torch.tensor(test_users, device=self.device)
        test_movies_torch = torch.tensor(test_movies, device=self.device)

        test_predictions = self.ncf_network(test_users_torch, test_movies_torch)
        reconstructed_matrix, _ = self.create_matrices(test_movies, test_users, test_predictions, default_replace='zero')

        predictions, index = self._extract_prediction_from_full_matrix(reconstructed_matrix, test_users,
                                                                       test_movies)
        predictions = self.postprocessing(predictions, postprocessing)
        if save_submission:
            self.save_submission(index, predictions, suffix=suffix)
        return predictions


class NCFNetwork(nn.Module):
    def __init__(self, number_of_users, number_of_movies, embedding_size):
        super().__init__()
        self.embedding_layer_users = nn.Embedding(number_of_users, embedding_size)
        self.embedding_layer_movies = nn.Embedding(number_of_movies, embedding_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(in_features=2 * embedding_size, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=1), # maybe predict per category?
            nn.ReLU()
        )

    def forward(self, users, movies):
        users_embedding = self.embedding_layer_users(users)
        movies_embedding = self.embedding_layer_movies(movies)
        concat = torch.cat([users_embedding, movies_embedding], dim=1)
        return torch.squeeze(self.feed_forward(concat))


def mse_loss(predictions, target):
    return torch.mean((predictions - target) ** 2)


def get_model(config, logger):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    ncf_network = NCFNetwork(config.NUM_USERS, config.NUM_MOVIES, params.embedding_size).to(device)
    return NCF(config, logger, ncf_network, device)
