from hex.pytorch.NNet import NNetWrapper as NNET
from hex.HexGame import HexGame as Game
from MCTS import MCTS
import numpy as np
from collections import deque
import sys
import os
import json
import pickle
import shutil
import subprocess
import time


class CoachAssistant():

    @staticmethod
    def distributeAndExecute(nnet, args):

        nnet.save_checkpoint('jobs', 'nnet.pth.tar')

        CoachAssistant.cleanEnvironments(args)

        subprocess.Popen(['sbatch', '-a', f'1-{args.numEps}', 'run_job.sh']).wait()

        
        time.sleep(3)

        while subprocess.check_output(['squeue', '-n', 'hex-training-job']).decode('utf-8').find('hex-trai') != -1:
            time.sleep(0.5)

        results = deque([])

        for job in range(1, args.numEps + 1):
            def error_exists():
                return os.path.exists('jobs/job_{0}/error.json'.format(job))
            def result_exists():
                return os.path.exists('jobs/job_{0}/result.bin'.format(job))
            
            while not error_exists() and not result_exists():
                time.sleep(0.5)            

            if error_exists():
                with open('jobs/job_{0}/error.json'.format(job)) as f:
                    error = f.read()
                    raise Exception('Job {0} failed: {1}'.format(job, error))

            with open('jobs/job_{0}/result.bin'.format(job), 'rb') as f:
                results.append(pickle.load(f))

        return results


    @staticmethod
    def cleanEnvironments(args):
        for jobNr in range(1, args.numEps + 1):
            if os.path.exists('jobs/job_{}/result.bin'.format(jobNr)):
                os.remove('jobs/job_{}/result.bin'.format(jobNr))
            if os.path.exists('jobs/job_{}/error.json'.format(jobNr)):
                os.remove('jobs/job_{}/error.json'.format(jobNr))

    @staticmethod
    def initEnvironments(args):
        if os.path.exists('jobs'):
            shutil.rmtree('jobs')
            while os.path.exists('jobs'):
                pass

        os.makedirs('jobs')
        for jobNr in range(1, args.numEps + 1):
            os.makedirs('jobs/job_{}'.format(jobNr))
            
        with open('jobs/args.json', 'w') as f:
            json.dump(args, f, indent=4)

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

        with open('jobs/args.json', 'r') as f:
            from utils import dotdict
            args = dotdict(json.load(f))

        game = Game(7)
        nnet = NNET(game)
        nnet.load_checkpoint('jobs', 'nnet.pth.tar')
        mcts = MCTS(game, nnet, args)

        trainExamples = []
        board = game.getInitBoard()
        curPlayer = 1
        episodeStep = 0

        print('starting MCTS search')

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
                assert valids[action] > 0

            board, _ = game.getNextState(canonicalBoard, 1, action)
            board = game.getOriginalForm(board, curPlayer)
            curPlayer = -curPlayer

            r = game.getGameEnded(board, curPlayer)

            if r != 0:
                result = [(x[0], x[2], r * ((-1) ** (x[1] != curPlayer)))
                          for x in trainExamples]

                with open('jobs/job_{}/result.bin'.format(jobNr), 'wb') as f:
                    pickle.dump(result, f)

                break


if __name__ == "__main__":

    jobNr = int(os.environ['SLURM_ARRAY_TASK_ID'])

    print('worker {} started'.format(jobNr))

    try:
        print('job {} started'.format(jobNr))
        CoachAssistant.executeEpisode()
        print('job {} finished'.format(jobNr))

    except Exception as e:
        with open('jobs/job_{}/error.json'.format(jobNr), 'w') as f:
            f.write(str(e))
    finally:
        print('job {} done'.format(jobNr))
