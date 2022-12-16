import json
import sys
import pickle

if len(sys.argv) == 1:
    raise Exception('No arguments passed')

worker = sys.argv[1]

# log_file = open('worker_{}/log.txt'.format(worker), 'w')

# def print(msg):
#     log_file.write(msg + '\n')

def do_work():
    
    import numpy as np

    from MCTS import MCTS
    from HexGame import HexGame as Game
    from NNet import NNetWrapper as NNET
    
    with open('args.json', 'r') as f:
        from utils import dotdict
        args = dotdict(json.load(f))
        
    board = np.load('worker_{0}/board.json.npy'.format(worker))
    
    game = Game(7)
    mcts = MCTS(game, NNET(game), args)
    _ = mcts.getActionProb(board, 1, temp=0)


    mcts_params = {
        'Nsa': mcts.Nsa
    }
    
    with open('worker_{0}/mcts.bin'.format(worker), 'wb') as f:
        pickle.dump(mcts_params, f)


if __name__ == "__main__":
    print('worker {} started'.format(worker))

    try:        
        do_work()
    except Exception as e:
        print('worker {} failed: {}'.format(worker, e))
        raise e
    finally:
        print('worker {} done'.format(worker))
