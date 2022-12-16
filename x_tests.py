from hex.HexGame import HexGame as Game
from hex.pytorch.NNet import NNetWrapper as NNET
from CoachAssistant import CoachAssistant
from utils import *

args = dotdict({
    'numIters': 10,
    'numEps': 1,
    'tempThreshold': 15,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 5,
    'arenaCompare': 2,
    'cpuct': 1,
    'numWorker': 10,

    'checkpoint': './pretrained_models/hex/pytorch/temp/',
    'load_model': False,
    'load_folder_file': ('./pretrained_models/hex/pytorch/dev/6x100','checkpoint_26.pth.tar'),
    'numItersForTrainExamplesHistory': 20
})

game = Game(7)
# mcts = MCTS(game, NNET(game), args)

# result = mcts.getActionProb(game.getInitBoard(), 1, temp=0)

# print(result)

nnet = NNET(game)
CoachAssistant.distributeAndExecute(nnet, args)