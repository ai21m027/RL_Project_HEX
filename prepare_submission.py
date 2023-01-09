import pickle
import time
import numpy as np
import logging
from HexPlayer import HexPlayer


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s (%(levelname)s) %(message)s')


player = HexPlayer(7, 100)
player.model.load_latest()

start = time.time()

np.random.seed(0)

for i in range(10):
    
    board = player.game.get_init_board()

    while not player.game.has_player2_won(board):
        pi = player.mcts.predict(board, 1)
        action = np.argmax(pi)
        player.game.make_move(board, action, 1)
        board = player.game.get_canonical_board(board, -1)

end = time.time()

print(end - start)

player.mcts.iterations = 10

pickle.dump(player, open('player.pkl', 'wb'))