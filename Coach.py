import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle
from hex.HexGame import display

import numpy as np
# from tqdm import tqdm

from Arena import Arena
from MCTS import MCTS as MCTS

from CoachAssistant import CoachAssistant

log = logging.getLogger(__name__)


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()
        
        CoachAssistant.initEnvironments(self.args)

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        start = 42
        for i in range(start, self.args.numIters + 1):
            # bookkeeping
            print(f'Starting Iter #{i} ...')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > start:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                for examples in CoachAssistant.distributeAndExecute(self.nnet, self.args):
                    iterationTrainExamples.extend(examples)
                    
                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iterationTrainExamples)

            while len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                print(f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            self.saveTrainExamples(i - 1)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(self.game, self.pnet, self.args)

            self.nnet.train(trainExamples)
            nmcts = MCTS(self.game, self.nnet, self.args)

            print('PITTING AGAINST PREVIOUS VERSION')
            def getNetworkAction(net, board, player):
                board = self.game.getCanonicalForm(board, player)
                valids2 = self.game.getValidMoves(board, 1)
                p, v = net.predict(board)
                p = p * valids2
                return np.random.choice(len(p), p=p / p.sum())
            
            def getMCTSAction(net, board, player):
                return np.argmax(net.getActionProb(board, player, temp=0))
            
            def getRandomAction(net, board, player):
                board = self.game.getCanonicalForm(board, player)
                valids2 = self.game.getValidMoves(board, 1)
                return np.random.choice(self.game.getActionSize(), p=valids2/valids2.sum())
            
            networkArena = Arena(lambda x, player: getNetworkAction(self.nnet, x, player),
                          lambda x, player: getNetworkAction(self.pnet, x, player), self.game, display=display)
            pwins, nwins, draws = networkArena.playGames(self.args.arenaCompare, verbose=False)            
            if pwins+nwins > 0 and float(nwins)/(pwins+nwins) < self.args.updateThreshold:
                print('Network - Bad results, Retesting with MCTS')
                mctsArena = Arena(lambda x, player: getMCTSAction(pmcts, x, player),
                            lambda x, player: getMCTSAction(nmcts, x, player), self.game, display=display)
                pwins, nwins, draws = mctsArena.playGames(self.args.arenaCompare, verbose=False)

            print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            # print('NEW/PREV WINS2 : %d / %d ; DRAWS : %d' % (nwins2, pwins2, draws2))
            if pwins+nwins > 0 and float(nwins)/(pwins+nwins) < self.args.updateThreshold:
                print('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                print('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            print(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            print("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            print('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
