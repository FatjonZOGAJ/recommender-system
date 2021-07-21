import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from lib.models.base_model import BaseModel
from lib.utils.utils import get_score


def loss_function_autoencoder(original, reconstructed, mask):
    return torch.mean(mask * (original - reconstructed) ** 2)


class Encoder(nn.Module):
    def __init__(self, input_dimension, encoded_dimension=250):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(in_features=input_dimension, out_features=500),
            nn.ReLU(),
            nn.Linear(in_features=500, out_features=encoded_dimension),
            nn.ReLU()
        )

    def forward(self, data):
        return self.model(data)


class Decoder(nn.Module):
    def __init__(self, output_dimensions, encoded_dimension=250):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(in_features=encoded_dimension, out_features=500),
            nn.ReLU(),
            nn.Linear(in_features=500, out_features=output_dimensions),
            nn.ReLU()
        )

    def forward(self, encodings):
        return self.model(encodings)


class AutoEncoderNetwork(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, data):
        return self.decoder(self.encoder(data))


class AutoEncoder(BaseModel):
    def __init__(self, autoencoder_network, num_users, num_movies, logger, device, learning_rate=1e-3, num_epochs=2000,
                 batch_size=64):

        super().__init__(logger)
        self.autoencoder_network = autoencoder_network
        self.logger = logger
        self.num_users = num_users
        self.num_movies = num_movies
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = device
        self.loss_function = loss_function_autoencoder
        self.optimizer = optim.Adam(self.autoencoder_network.parameters(), lr=learning_rate)

    def fit(self, train_movies, train_users, train_predictions, **kwargs):
        # Build Dataloaders
        data, mask = self.create_matrices(train_movies, train_users, train_predictions)
        self.data_torch = torch.tensor(data, device=self.device).float()
        self.mask_torch = torch.tensor(mask, device=self.device)

        dataloader = DataLoader(
            TensorDataset(self.data_torch, self.mask_torch),
            batch_size=self.batch_size)
        step = 0
        for epoch in range(self.num_epochs):
            for data_batch, mask_batch in dataloader:
                self.optimizer.zero_grad()

                reconstructed_batch = self.autoencoder_network(data_batch)

                loss = self.loss_function(data_batch, reconstructed_batch, mask_batch)

                loss.backward()

                self.optimizer.step()
                step += 1

            if epoch % 5 == 0:
                reconstructed_matrix = self.reconstruct_whole_matrix()
                predictions = self._extract_prediction_from_full_matrix(reconstructed_matrix, kwargs['val_users'],
                                                                        kwargs['val_movies'], save_submission=False,
                                                                        transpose_matrix=True)
                reconstruction_rmse = get_score(predictions, kwargs['val_predictions'])
                self.logger.info('At epoch {:3d} loss is {:.4f}'.format(epoch, reconstruction_rmse))

    def predict(self, test_movies, test_users, save_submission):
        reconstructed_matrix = self.reconstruct_whole_matrix()
        return self._extract_prediction_from_full_matrix(reconstructed_matrix, test_users,
                                                         test_movies, save_submission=save_submission,
                                                         transpose_matrix=True)

    def reconstruct_whole_matrix(self):
        data_reconstructed = np.zeros((self.num_users, self.num_movies))

        with torch.no_grad():
            for i in range(0, self.num_users, self.batch_size):
                upper_bound = min(i + self.batch_size, self.num_users)
                data_reconstructed[i:upper_bound] = self.autoencoder_network(self.data_torch[i:upper_bound]).detach().cpu().numpy()

        return data_reconstructed


def get_model(config, logger):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder_network = AutoEncoderNetwork(
        encoder=Encoder(
            input_dimension=config.NUM_MOVIES,
            encoded_dimension=config.ENCODED_DIMENSION,
        ),
        decoder=Decoder(
            output_dimensions=config.NUM_MOVIES,
            encoded_dimension=config.ENCODED_DIMENSION,
        )
    ).to(device)
    return AutoEncoder(autoencoder_network, config.NUM_USERS, config.NUM_MOVIES, logger, device)