from MCTS import MCTS
from hex.HexGame import HexGame
from hex.pytorch.NNet import NNetWrapper as nn
import numpy as np
from utils import dotdict

class HexPlayer():
    def __init__(self, nnet_directory, nnet_file):
        args = dotdict({
            'numMCTSSims': 50,
            'cpuct': 1
        })

        self.n = 7
        self.game = HexGame(self.n)
        self.nnet = nn(self.game)
        self.nnet.load_checkpoint(nnet_directory, nnet_file)
        self.mcts = MCTS(self.game, self.nnet, args)

    def play(self, board, player):
        
        player = 1 if player == 1 else -1

        def transformAction(action):
            x = action // self.n
            y = action % self.n
            
            board = np.zeros(shape=(self.n, self.n))
            board[x][y] = 1

            board = self.game.getOriginalForm(board, player)
            loc = np.where(board == player)
            x, y = loc[0][0], loc[1][0]
            return (x, y)

        cannoncial_action = np.argmax(self.mcts.getActionProb(np.array(board), player, temp=0))

        return transformAction(cannoncial_action)
    

from hex.hex_engine import hexPosition



def savePlayer():
    import pickle
    
    nnet_player = HexPlayer('./pretrained_models/hex/pytorch','checkpoint_261.pth.tar')

    with open('hex_player.pkl', 'wb') as f:
        pickle.dump(nnet_player, f)
        
# savePlayer()

def loadPlayer():    
    import pickle
    with open('hex_player.pkl', 'rb') as f:
        return pickle.load(f)
    

nnet_player = loadPlayer()
game = hexPosition(7)
game.humanVersusMachine(machine=lambda board, _: nnet_player.play(board, 2))