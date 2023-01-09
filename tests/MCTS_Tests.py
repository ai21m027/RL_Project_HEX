import math
import unittest
import numpy as np

from HexGame import HexGame
from HexNet import HexNet
from MCTS import MCTS


class MCTS_Tests(unittest.TestCase):

    def setUp(self):
        self.game = HexGame(4)

    def test_weak_mcts_vs_strong_mcts(self):

        self.execute_games(
            nGames=100,
            expectedWinner=-1,
            player1=self.create_mcts_player(2),
            player2=self.create_mcts_player(10)
        )

    def test_mcts_vs_trained_model(self):

        self.execute_games(
            nGames=100,
            expectedWinner=1,
            player1=self.create_mcts_player(100),
            player2=self.create_mcts_player(100, load_latest_model=True)
        )

    def test_random_vs_trained_model(self):

        self.execute_games(
            nGames=100,
            expectedWinner=-1,
            player1=self.create_random_player(),
            player2=self.create_mcts_player(10, load_latest_model=True)
        )

    def test_mcts_vs_random(self):

        self.execute_games(
            nGames=100,
            expectedWinner=1,
            player1=self.create_mcts_player(100),
            player2=self.create_random_player()
        )

    def test_random_vs_random(self):

        self.execute_games(
            nGames=100,
            expectedWinner=None,
            player1=self.create_random_player(),
            player2=self.create_random_player()
        )

    def execute_games(self, nGames, expectedWinner, player1, player2):

        winners = {1: 0, -1: 0}
        players = {1: player1, -1: player2}

        switch_player = False
        for i in range(nGames):

            board = self.game.get_init_board()
            playerId = 1

            while True:
                player = players[-playerId] if switch_player else players[playerId]

                x, y = player(board)
                self.game.make_move_xy(board, x, y, 1)

                winner = playerId if self.game.has_player1_won(board) else 0
                winner = -winner if switch_player else winner

                if winner != 0:
                    print("Winner is player", winner)
                    winners[winner] += 1
                    break

                board = self.game.get_canonical_board(board, -1)

                playerId = -playerId

        if expectedWinner != None:
            self.assertTrue(winners[expectedWinner] > winners[-expectedWinner],
                            "Expected player " + str(expectedWinner) +
                            " to win more games than player " + str(-expectedWinner) +
                            " but player " + str(expectedWinner) +
                            " won " + str(winners[expectedWinner]) +
                            " times and player " + str(-expectedWinner) +
                            " won " + str(winners[-expectedWinner]) + " times.")

    def create_mcts_player(self, iterations, model_path=None, load_latest_model=False):

        nnet = HexNet(self.game, use_cuda=False)
        if model_path is not None:
            nnet.load(model_path)
        elif load_latest_model:
            nnet.load_latest()
        mcts = MCTS(self.game, nnet, iterations, cpuct=1.0)

        def play(board):
            action = np.argmax(mcts.predict(board, 1))
            return self.game.get_original_action(action, 1)

        return play

    def create_random_player(self):

        def play(board):
            rn = np.random.rand(self.game.action_size) * self.game.get_action_mask(board)
            action = np.argmax(rn)
            x, y = self.game.get_original_action(action, 1)
            return x, y

        return play
