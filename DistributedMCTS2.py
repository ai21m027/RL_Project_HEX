import os
import json
import pickle
import shutil
import subprocess
import numpy as np

class DistributedMCTS2():

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args

        self.worker = 10
        
        if os.path.exists('shared'):
            shutil.rmtree('shared')
            while os.path.exists('shared'):
                pass
            
        os.makedirs('shared')
        for i in range(self.worker):
            os.makedirs('shared/worker_{}'.format(i))

        with open('shared/args.json', 'w') as f:
            json.dump(self.args, f, indent=4)
        
        shutil.copyfile('hex/pytorch/NNet.py', 'shared/NNet.py')
        shutil.copyfile('hex/pytorch/HexNNet.py', 'shared/HexNNet.py')
        shutil.copyfile('MCTS.py', 'shared/MCTS.py')
        shutil.copyfile('DistributedMCTSClient.py', 'shared/DistributedMCTSClient.py')
        shutil.copyfile('Game.py', 'shared/Game.py')
        shutil.copyfile('utils.py', 'shared/utils.py')
        shutil.copyfile('hex/HexGame.py', 'shared/HexGame.py')
        shutil.copyfile('hex/HexLogic.py', 'shared/HexLogic.py')
        shutil.copyfile('RunClient.ps1', 'shared/RunClient.ps1')

        
    def getActionProb(self, board, player, temp=1):

        canonicalBoard = self.game.getCanonicalForm(board, player)
        game_ended = self.game.getGameEnded(canonicalBoard, 1)
        if game_ended != 0:
            return -game_ended
        
        self.nnet.save_checkpoint('shared', 'nnet.pth.tar')

            
        valids = [x[0] for x in enumerate(self.game.getValidMoves(canonicalBoard, 1))]
        probablilities, v = self.nnet.predict(canonicalBoard)
        
        sorted_actions = list(filter(lambda x: x in valids, [x[0] for x in sorted(enumerate(probablilities), key=lambda x: x[1])]))
        
        processes = []

        for i in range(min(10, len(valids))):
            if i < len(sorted_actions):
                new_board, _ = self.game.getNextState(canonicalBoard, 1, sorted_actions[i])
                
                np.save('shared/worker_{0}/board.json'.format(i), new_board)

                print('starting worker {}'.format(i))
                process = subprocess.Popen(['powershell', './RunClient.ps1', str(i)], cwd=r'shared')
                processes.append(process)
        
        success = True

        global_nsa = {}

        for worker, process in enumerate(processes):
            if process.wait() != 0:
                success = False
            else:                
                with open('shared/worker_{0}/mcts.bin'.format(worker), 'rb') as f:
                    mcts_results = pickle.load(f)
                    
                    nsa = mcts_results['Nsa']

                    for key in nsa:
                        if key not in global_nsa:
                            global_nsa[key] = 0
                        global_nsa[key] += nsa[key]


        if not success:
            raise Exception('worker failed')
        
        s = self.game.stringRepresentation(canonicalBoard)
        counts = [global_nsa[(s,a)] if (s,a) in global_nsa else 0 for a in range(self.game.getActionSize())]

        if temp==0:
            bestA = np.argmax(counts)
            probs = [0]*len(counts)
            probs[bestA]=1
            return probs

        counts = [x**(1./temp) for x in counts]
        probs = [x/float(sum(counts)) for x in counts]
        return probs        