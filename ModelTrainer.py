import logging
from collections import deque
from datetime import datetime
import numpy as np
from utils import *
import glob

from HexGame import HexGame
from HexNet import HexNet
from MCTS import MCTS


class ModelTrainer:
    def __init__(self, game: HexGame, num_upgrades: int, max_training_data: int, num_self_play: int, self_play_mcts_sims: int) -> None:
        self.game = game
        self.num_upgrades = num_upgrades
        self.max_training_data = max_training_data
        self.num_self_play = num_self_play
        self.self_play_mcts_sims = self_play_mcts_sims
        self.last_data_file_check_time = datetime.min
        self.acceptance_threshold = 0.6

    def run(self, load_latest_model: bool):
        model = HexNet(self.game, use_cuda=True)
        if load_latest_model:
            model.load_latest()

        model_copy = model.copy()

        training_boards = deque(maxlen=self.max_training_data)
        training_pis = deque(maxlen=self.max_training_data)
        training_values = deque(maxlen=self.max_training_data)

        iteration = 1
        upgrades = 0

        while upgrades < self.num_upgrades:
            for boards, pis, values in self.__load_new_data():
                training_boards.extend(boards)
                training_pis.extend(pis)
                training_values.extend(values)

            if len(training_boards) == 0:
                logging.info(f"Iteration {iteration}: No training data found")
                time.sleep(10)
                continue

            logging.info(f"Starting training iteration {iteration} with data size: {len(training_boards)}")

            model.change_device(use_cuda=True)
            measure_time(model.train)(training_boards, training_pis, training_values)
            model.save_temp()

            if measure_time(self.__evaluate_model)(model_copy, model):
                upgrades += 1
                logging.info(f"Iteration {iteration}: Model upgrade {upgrades} accepted")
                model.save()
                model_copy = model.copy()
            else:
                logging.info(f"Iteration {iteration}: New model rejected")
                # model = model_copy
                

            iteration += 1

    def __evaluate_model(self, prev_model: HexNet, new_model: HexNet):
        prev_wins = 0
        new_wins = 0

        prev_model.change_device(use_cuda=False)
        new_model.change_device(use_cuda=False)

        mcts_prev = MCTS(self.game, prev_model, self.self_play_mcts_sims)
        mcts_new = MCTS(self.game, new_model, self.self_play_mcts_sims)

        for i in range(self.num_self_play):
            if self.__play_game(mcts_new, mcts_prev, switch_player=((i % 2) == 1)) == 1:
                new_wins += 1
            else:
                prev_wins += 1

        logging.info(f"Evaluation Prev: {prev_wins}, New: {new_wins}")

        return new_wins / self.num_self_play > self.acceptance_threshold

    def __play_game(self, mcts1: MCTS, mcts2: MCTS, switch_player: bool):
        board = self.game.get_init_board()

        player = 1
        players = {1: mcts1, -1: mcts2} if not switch_player else {1: mcts2, -1: mcts1}
        while True:

            pi = players[player].predict(board, 1)
            action = np.argmax(pi)
            self.game.make_move(board, action, 1)

            if self.game.has_player1_won(board):
                return -player if switch_player else player

            board = self.game.get_canonical_board(board, -1)
            player *= -1

    def __load_new_data(self):

        worker_paths = glob.glob("data/worker_*")

        min_file_name = f'{self.last_data_file_check_time.strftime("%Y%d%m_%H%M%S")}.npz'
        self.last_data_file_check_time = datetime.now()

        total_files = 0
        total_size = 0

        for worker_path in worker_paths:
            worker_files = filter(lambda file: file > min_file_name, glob.glob(f"{worker_path}/*.npz"))
            for worker_file in worker_files:                
                try:
                    data = np.load(worker_file)
                    boards = data["boards"]
                    pis = data["pis"]
                    values = data["values"]

                    yield boards, pis, values

                    total_files += 1
                    total_size += len(boards)
                    data.close()
                except Exception as e:
                    logging.error(f"Error loading data from {worker_file}: {e} - Deleting File")  
                
                os.remove(worker_file)

        logging.info(f"Loaded {total_files} files with {total_size} training data")



if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s (%(levelname)s) %(message)s')

    game = HexGame(7)
    trainer = ModelTrainer(game, num_upgrades=1000, max_training_data=200000, num_self_play=50, self_play_mcts_sims=10)

    trainer.run(load_latest_model=True)