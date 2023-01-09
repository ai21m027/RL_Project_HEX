import logging
import math
from datetime import datetime
import glob
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from HexGame import HexGame as Game
from utils import file_write_atomic


class HexNetModel(nn.Module):
    def __init__(self, game: Game) -> None:
        super(HexNetModel, self).__init__()
        self.game = game

        self.num_layers = 64

        self.conv1 = nn.Conv2d(1, self.num_layers, 3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(self.num_layers)
        self.conv2 = nn.Conv2d(self.num_layers, self.num_layers, 3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(self.num_layers)
        self.conv3 = nn.Conv2d(self.num_layers, self.num_layers, 3, padding=1)
        self.conv3_bn = nn.BatchNorm2d(self.num_layers)
        self.fc1 = nn.Linear(self.num_layers * self.game.size * self.game.size, 512)
        self.fc1_bn = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.fc2_bn = nn.BatchNorm1d(256)
        self.fc_pi = nn.Linear(256, game.action_size)
        self.fc_v = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = x.view(-1, self.num_layers * self.game.size * self.game.size)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.relu(self.fc2_bn(self.fc2(x)))
        pi = self.fc_pi(x)
        v = self.fc_v(x)
        return F.log_softmax(pi, dim=1), torch.tanh(v)


class HexNet:
    def __init__(self, game: Game, use_cuda: bool) -> None:
        self.game = game
        self.use_cuda = use_cuda
        self.device = torch.device('cuda') if use_cuda and torch.cuda.is_available() else torch.device('cpu')
        self.model = HexNetModel(game).to(self.device)
        self.epochs = 20
        self.batch_size = 1024
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.policy_loss = nn.CrossEntropyLoss()
        self.value_loss = nn.MSELoss()
        self.model.eval()

    def change_device(self, use_cuda: bool):
        self.use_cuda = use_cuda
        self.device = torch.device('cuda') if use_cuda and torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)

    def predict(self, board: np.array):
        board = torch.tensor(board, dtype=torch.float32, device=self.device)
        
        with torch.no_grad(): 
            pi, v = self.model(board.view(1, 1, self.game.size, self.game.size))
        return pi.exp().data.cpu().numpy()[0], v.data.cpu().item()

    def train(self, boards, pis, wins):
        assert len(boards) > 0 and len(boards) == len(pis) and len(boards) == len(wins)

        self.model.train()

        for _ in range(self.epochs):

            epoch_loss = 0
            batches = math.ceil(len(boards) / self.batch_size)

            for _ in range(batches):
                sample_ids = np.random.randint(0, len(boards), self.batch_size)
                boards_batch = np.array([boards[i] for i in sample_ids])
                pis_batch = np.array([pis[i] for i in sample_ids])
                wins_batch = np.array([wins[i] for i in sample_ids])
                loss = self.__train_on_batch(boards_batch, pis_batch, wins_batch)
                epoch_loss += loss
                
            epoch_loss /= batches
            # print('Epoch loss: ', epoch_loss)
            
        self.model.eval()

    def copy(self):
        net = HexNet(self.game, self.use_cuda)
        net.model.load_state_dict(self.model.state_dict())
        return net

    def save_temp(self, keep: int = 5):
        if not os.path.isdir('models'):
            os.mkdir('models')

        n = self.game.size
        file = f'models/model_{n}x{n}.{datetime.now().strftime("%Y%d%m_%H%M%S")}.current.pth'
        logging.info(f'Saving model to {file}')
        file_write_atomic(file, lambda f: torch.save(self.model.state_dict(), f))
        
        files = self.get_saved_model_files()

        for file in sorted(files, reverse=True):
            if file.endswith('.current.pth'):
                if keep > 0:
                    keep -= 1
                else:
                    os.remove(file)

    def save(self, keep: int = 5):
        if not os.path.isdir('models'):
            os.mkdir('models')

        n = self.game.size
        file = f'models/model_{n}x{n}.{datetime.now().strftime("%Y%d%m_%H%M%S")}.pth'
        logging.info(f'Saving model to {file}')
        file_write_atomic(file, lambda f: torch.save(self.model.state_dict(), f))
        
        files = self.get_saved_model_files()

        for file in files[:-keep]:
            os.remove(file)

    def load_latest(self, path: str = None):
        files = self.get_saved_model_files(path)
        if len(files) > 0:
            self.load(files[-1])

    def get_saved_model_files(self, path: str = None):
        if path is None:
            path = 'models'

        if not os.path.isdir(path):
            logging.error('Models directory not found.')
            return

        n = self.game.size
        return sorted(glob.glob(f'{path}/model_{n}x{n}.*.pth'))

    def load(self, path):
        if not os.path.isfile(path):
            logging.error(f'Could not load model from: {path}. File does not exist.')
            return
        logging.info(f'Loading model from {path}')
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def __train_on_batch(self, boards, pis, wins):
        boards = torch.tensor(boards, dtype=torch.float32, device=self.device)
        pis = torch.tensor(pis, dtype=torch.float32, device=self.device)
        wins = torch.tensor(wins, dtype=torch.float32, device=self.device)

        self.optimizer.zero_grad()
        pi, v = self.model(boards.view(-1, 1, self.game.size, self.game.size))
        pi_loss = self.policy_loss(pi, pis.view(-1, self.game.action_size))
        v_loss = self.value_loss(v, wins.view(-1, 1))
        total_loss = pi_loss + v_loss
        
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()
