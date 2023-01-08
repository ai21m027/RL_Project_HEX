import logging
import os
import sys
import numpy as np
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle
from hex.HexGame import display
from utils import *

from Arena import Arena
from MCTS import MCTS as MCTS
from CoachAssistant import CoachAssistant


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

        for iteration in range(self.args.startIteration, self.args.numIters + 1):
            self.runIteration(iteration)

    def runIteration(self, iteration):
            # bookkeeping
            logging.info(f'Starting Iter #{iteration} ...')

            while True:
                measure_time(self.createTrainingData)(iteration)

                measure_time(self.trainNetwork)()

                if measure_time(self.evaluateNetwork)():
                    logging.warn('REJECTING NEW MODEL and start RE-TRAINING')
                    self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
                else:
                    logging.info('ACCEPTING NEW MODEL')
                    self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(iteration))
                    self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
                    break


    def createTrainingData(self, iteration):
        if not self.skipFirstSelfPlay or iteration > self.args.startIteration:
            iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

            if self.args.use_cluster_sampling:
                for examples in CoachAssistant.distributeAndExecute(self.nnet, self.args):
                    iterationTrainExamples.extend(examples)
            else:
                for _ in range(self.args.numEps):
                    self.mcts = MCTS(self.game, self.nnet, self.args)  # reset search tree
                    iterationTrainExamples += self.executeEpisode()
                
            # save the iteration examples to the history 
            self.trainExamplesHistory.append(iterationTrainExamples)

        while len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
            logging.info(f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
            self.trainExamplesHistory.pop(0)

        # backup history to a file
        # deacitvated for now as it is not necessary and takes a lot of time
        # measure_time(self.saveTrainExamples, f'save {np.sum([len(x) for x in self.trainExamplesHistory])} training examples')(iteration - 1)

    def trainNetwork(self):
        # shuffle examples before training
        trainExamples = []
        for e in self.trainExamplesHistory:
            trainExamples.extend(e)
        shuffle(trainExamples)

        # training new network, keeping a copy of the old one
        self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
        self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
        
        self.nnet.train(trainExamples)

    def evaluateNetwork(self):

        pmcts = MCTS(self.game, self.pnet, self.args)
        nmcts = MCTS(self.game, self.nnet, self.args)

        print('PITTING AGAINST PREVIOUS VERSION')        
        def getMCTSAction(net, board, player):
            return np.argmax(net.getActionProb(board, player, temp=0))
        
        mctsArena = Arena(lambda x, player: getMCTSAction(pmcts, x, player),
                    lambda x, player: getMCTSAction(nmcts, x, player), self.game, display=display)
        
        pwins, nwins, draws = mctsArena.playGames(self.args.arenaCompare, verbose=False)
        
        logging.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
        return pwins+nwins > 0 and float(nwins)/(pwins+nwins) < self.args.updateThreshold




    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)

            pi = self.mcts.getActionProb(canonicalBoard, self.curPlayer, temp=temp)
            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, self.curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            
            valids = self.game.getValidMoves(board, self.curPlayer)

            if valids[action] == 0:
                print('invalid action in coach', action)
                assert valids[action] >0   
                
            board, _ = self.game.getNextState(canonicalBoard, 1, action)
            board = self.game.getOriginalForm(board, self.curPlayer)
            self.curPlayer = -self.curPlayer

            r = self.game.getGameEnded(board, self.curPlayer)

            if r != 0:
                return [(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer))) for x in trainExamples]





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
