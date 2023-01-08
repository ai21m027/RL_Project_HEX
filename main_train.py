from Coach import Coach
from hex.HexGame import HexGame as Game
from hex.pytorch.NNet import NNetWrapper as nn
from utils import *
import logging

logging.basicConfig(filename='logs/log.txt', encoding='utf-8', level=logging.DEBUG, format='%(asctime)s (%(levelname)s) %(message)s')

args = dotdict({
    'numIters': 1000,
    'numEps': 100,
    'tempThreshold': 15,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 25,
    'arenaCompare': 20,
    'cpuct': 1,

    'checkpoint': './pretrained_models/hex/pytorch/temp/',
    'load_model': True,
    'load_folder_file': ('./pretrained_models/hex/pytorch','checkpoint_361.pth.tar'),
    'startIteration': 362,
    'numItersForTrainExamplesHistory': 10,

    'use_cluster_sampling': True,
    'num_workers': 15,
    'numEps_per_worker': 10
})

if __name__=="__main__":
    logging.info('Starting training')

    g = Game(7)
    nnet = nn(g)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    # if args.load_model:
    #     logging.info("Load trainExamples from file")
    #     c.loadTrainExamples()
    c.learn()
