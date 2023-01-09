import unittest
import numpy as np
import torch
from HexGame import HexGame
from HexNet import HexNet

class HexNet_Tests(unittest.TestCase):

    def setUp(self) -> None:
        self.game = HexGame(3)
        self.model = HexNet(self.game, use_cuda=False)

    def test_train_model(self):
        board = self.game.get_init_board()

        boards = [board]
        pis = [[0, 1, 0, 0, 0, 0, 0, 0, 0]]
        wins = [1]
        
        self.model.train(boards, pis, wins)

        pi, v = self.model.predict(board)

        self.assertEqual(np.argmax(pi), 1)
        self.assertTrue(v > 0)

    def test_train_model2(self):
        board = self.game.get_init_board()

        boards = [board]
        pis = [[0, 0, 0, 0, 0, 1, 0, 0, 0]]
        wins = [-1]
        
        self.model.train(boards, pis, wins)

        pi, v = self.model.predict(board)

        self.assertEqual(np.argmax(pi), 5)
        self.assertTrue(v < 0)