import numpy as np
from HexGame import HexGame
from HexNet import HexNet
from MCTS import MCTS


class HexPlayer:
    def __init__(self, game_size: int, mcts_steps: int):
        self.game = HexGame(game_size)
        self.model = HexNet(self.game, use_cuda=False)
        self.mcts = MCTS(self.game, self.model, mcts_steps)

    def select_move(self, board, player, randomize=False):

        board = np.array(board)
        assert board.shape == (self.game.size, self.game.size)
        assert player == 1 or player == 2

        if player == 2:
            player = -1

        board = np.array([1 if x == 1 else -1 if x == 2 else 0 for x in board.ravel()]).reshape(self.game.size, self.game.size)

        pi = self.mcts.predict(board, player)
        action = np.random.choice(len(pi), p=pi) if randomize else np.argmax(pi)
        x, y = self.game.get_original_action(action, player)

        return y, x
