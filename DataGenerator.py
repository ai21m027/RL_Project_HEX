from datetime import datetime
import glob
import logging
import numpy as np
import os
from utils import *
from HexNet import HexNet
from HexGame import HexGame
from MCTS import MCTS


class DataGenerator:

    def __init__(self, game: HexGame, worker_id: int, num_episodes: int, num_mcts_sims: int, max_data_files: int) -> None:
        self.worker_id = worker_id
        self.game = game
        self.model = HexNet(self.game, use_cuda=False)
        self.num_episodes = num_episodes
        self.num_mcts_sims = num_mcts_sims
        self.max_data_files = max_data_files

    def run(self) -> None:
        while True:
            measure_time(self.__generate_data)()

    def __generate_data(self):
        self.model.change_device(use_cuda=False)

        mcts = MCTS(self.game, self.model, self.num_mcts_sims)

        for _ in range(self.num_episodes):
            self.model.load_latest()
            boards, pis, values = self.__play_episode(mcts)
            self.__save_data(boards, pis, values)

            logging.info(f"Generated data size: {len(boards)}")

    def __play_episode(self, mcts: MCTS):

        episode_boards = []
        episode_pis = []
        episode_values = []

        # we play always as player 1 with the canonical board
        # then player variable is only used to generate the values
        player = 1
        board = self.game.get_init_board()

        while True:
            pi = mcts.predict(board, 1)
            action = np.random.choice(self.game.action_size, p=pi)
            self.game.make_move(board, action, 1)

            if self.game.has_player1_won(board):
                if player == -1:
                    episode_values = [v * -1 for v in episode_values]
                return episode_boards, episode_pis, episode_values
            
            boards, pis = self.game.get_symmetries(board, pi)
            episode_boards.extend(boards)
            episode_pis.extend(pis)            
            episode_values.extend([player] * len(boards))

            board = self.game.get_canonical_board(board, -1)
            player *= -1

    def __save_data(self, boards, pis, values):

        while len(self.__get_saved_data_files()) >= self.max_data_files:
            logging.info("Data folder is full, waiting for 10 seconds...")
            time.sleep(10)

        boards = np.array(boards)
        pis = np.array(pis)
        values = np.array(values)

        worker_path = self.__get_worker_path()

        file_name = f'{worker_path}/{datetime.now().strftime("%Y%d%m_%H%M%S")}.npz'
        file_write_atomic(file_name, worker_id=self.worker_id, callback=lambda f: np.savez(f, boards=boards, pis=pis, values=values))

    def __get_saved_data_files(self):
        worker_path = self.__get_worker_path()
        return sorted(glob.glob(f'{worker_path}/*.npz'))

    def __get_worker_path(self):
        worker_path = f"data/worker_{self.worker_id}"

        if not os.path.exists(worker_path):
            os.makedirs(f"data/worker_{self.worker_id}")
        return worker_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s (%(levelname)s) %(message)s')

    if os.environ.__contains__('SLURM_ARRAY_TASK_ID'):
        workerId = int(os.environ['SLURM_ARRAY_TASK_ID'])
    else:
        workerId = 0
    
    game = HexGame(7)
    generator = DataGenerator(game, workerId, num_episodes=200, num_mcts_sims=200, max_data_files=100)
    generator.run()
