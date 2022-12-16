from collections import deque
import sys
import os
import json
import pickle
import shutil
import subprocess
from queue import Queue
sys.path.append('hex')
sys.path.append('hex/pytorch')

import numpy as np
from MCTS import MCTS
from HexGame import HexGame as Game
from NNet import NNetWrapper as NNET

class CoachAssistant():

    @staticmethod
    def distributeAndExecute(nnet, args):
                
        CoachAssistant.prepareEnvironments(nnet, args)

        worker = args.numWorker
        jobs = list(range(args.numEps))

        running_jobs = {}       
        results = deque([])

        def popNextFinishedJob():
            for job in running_jobs:
                result = running_jobs[job]()
                if result is not None:
                    del running_jobs[job]
                    return result
            return []



        while len(jobs) > 0:
            if(len(running_jobs) < worker):
                job = jobs.pop(0)
                running_jobs[job] = CoachAssistant.startJob(job, worker)
            else:                
                results += popNextFinishedJob()

        while len(running_jobs) > 0:
            results += popNextFinishedJob()

        return results

    @staticmethod
    def startJob(job, worker):
        
        def getResult():

            if os.path.exists('shared/jobs/job_{0}/failed'.format(job)):
                with open('shared/jobs/job_{0}/error.json'.format(job)) as f:
                    error = f.read()
                    raise Exception('Job {0} failed: {1}'.format(job, error))

            if os.path.exists('shared/jobs/job_{0}/done'.format(job)):
                with open('shared/jobs/job_{0}/result.bin'.format(job), 'rb') as f:
                    return pickle.load(f)
            else:
                return None

        
        subprocess.Popen(['powershell', './RunClient.ps1', str(job)], cwd=r'shared', creationflags=subprocess.CREATE_NEW_CONSOLE)

        return getResult

    @staticmethod
    def prepareEnvironments(nnet, args):

        CoachAssistant.createDirectories(args.numEps)

        with open('shared/args.json', 'w') as f:
            json.dump(args, f, indent=4)

        nnet.save_checkpoint('shared', 'nnet.pth.tar')
        
        shutil.copyfile('hex/pytorch/NNet.py', 'shared/NNet.py')
        shutil.copyfile('hex/pytorch/HexNNet.py', 'shared/HexNNet.py')
        shutil.copyfile('MCTS.py', 'shared/MCTS.py')
        shutil.copyfile('CoachAssistant.py', 'shared/CoachAssistant.py')
        shutil.copyfile('Game.py', 'shared/Game.py')
        shutil.copyfile('utils.py', 'shared/utils.py')
        shutil.copyfile('hex/HexGame.py', 'shared/HexGame.py')
        shutil.copyfile('hex/HexLogic.py', 'shared/HexLogic.py')
        shutil.copyfile('RunClient.ps1', 'shared/RunClient.ps1')

    @staticmethod
    def createDirectories(nJobs):
        if os.path.exists('shared'):
            shutil.rmtree('shared')
            while os.path.exists('shared'):
                pass
            
        os.makedirs('shared/jobs')
        for i in range(nJobs):
            os.makedirs('shared/jobs/job_{}'.format(i))
    



    @staticmethod
    def executeEpisode():
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

        
        with open('args.json', 'r') as f:
            from utils import dotdict
            args = dotdict(json.load(f))

        game = Game(7)
        nnet = NNET(game)
        nnet.load_checkpoint('.', 'nnet.pth.tar')
        mcts = MCTS(game, nnet, args)

        trainExamples = []
        board = game.getInitBoard()
        curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard = game.getCanonicalForm(board, curPlayer)
            temp = int(episodeStep < args.tempThreshold)

            pi = mcts.getActionProb(canonicalBoard, curPlayer, temp=temp)
            sym = game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            
            valids = game.getValidMoves(board, curPlayer)

            if valids[action] == 0:
                print('invalid action in coach', action)
                assert valids[action] >0   
                
            board, _ = game.getNextState(canonicalBoard, 1, action)
            board = game.getOriginalForm(board, curPlayer)
            curPlayer = -curPlayer

            r = game.getGameEnded(board, curPlayer)

            if r != 0:
                result = [(x[0], x[2], r * ((-1) ** (x[1] != curPlayer))) for x in trainExamples]
                
                with open('jobs/job_{}/result.bin'.format(jobNr), 'wb') as f:
                    pickle.dump(result, f)

                break
            



            
if __name__ == "__main__":
    
    if len(sys.argv) == 1:
        raise Exception('No arguments passed')

    jobNr = sys.argv[1]

    print('worker {} started'.format(jobNr))

    try:
        CoachAssistant.executeEpisode()

        open('jobs/job_{}/done'.format(jobNr), 'w').close()

    except Exception as e:
        print('job {} failed: {}'.format(jobNr, e))
        
        with open('jobs/job_{}/error.json'.format(jobNr), 'w') as f:
            f.write(str(e))

        open('jobs/job_{}/failed'.format(jobNr), 'w').close()
    finally:
        print('job {} done'.format(jobNr))