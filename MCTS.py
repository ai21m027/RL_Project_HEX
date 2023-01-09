import logging
import math
import numpy as np
from HexGame import HexGame as Game
from HexNet import HexNet

EPS = 1e-8


class MCTS_State:
    def __init__(self) -> None:
        self.winner = None
        self.PI = None
        self.Na = None
        self.Qa = None
        self.N = 0


class MCTS:
    def __init__(self, game: Game, model: HexNet, iterations: int) -> None:
        self.game = game
        self.model = model
        self.iterations = iterations
        self.states = {}

    def predict(self, board: Game, player: int):

        canonical = self.game.get_canonical_board(board, player)

        if self.iterations == 0:
            pi, _ = self.model.predict(canonical)
            pi *= self.game.get_action_mask(canonical)
            sum = np.sum(pi)
            if sum > 0:
                pi /= sum
            else:
                logging.warning("All valid moves were masked, do workaround.")
                pi = np.ones(self.game.action_size) / self.game.action_size
            return pi


        for _ in range(self.iterations):
            self.__search(canonical, 1)

        state = self.__get_state(canonical)

        return state.Na / state.N

    def __search(self, board: np.array, player: int):

        canonical = self.game.get_canonical_board(board, player)
        state = self.__get_state(canonical)

        if state.winner != 0:
            # previous player made last move so we need to invert the winner
            return -state.winner

        if (state.PI is None):
            state.PI, value = self.model.predict(canonical)
            state.PI *= self.game.get_action_mask(canonical)
            sum = np.sum(state.PI)
            if sum  > 0:
                state.PI /= sum
            else:
                logging.warning("All valid moves were masked, do workaround.")
                state.PI = np.ones(self.game.action_size) / self.game.action_size
            return -value

        action = self.__select_action(state, canonical)
        self.game.make_move(canonical, action, 1)

        value = self.__search(canonical, -1)
        self.__update_state(state, action, value)

        return -value

    def __get_state(self, canonical: np.array):
        s = self.game.get_state_idx(canonical)
        state = self.states.get(s)

        if state is None:
            state = MCTS_State()
            # state is from previous player perspective, so we need to invert the winner
            state.winner = -1 if self.game.has_player2_won(canonical) else 0
            assert self.game.has_player1_won(canonical) == False
            state.Na = np.zeros(self.game.action_size, dtype=int)
            state.Qa = np.zeros(self.game.action_size, dtype=float)
            self.states[s] = state
        return state

    def __update_state(self, state: MCTS_State, action: int, winner: int):
        if state.Na[action] == 0:
            state.Qa[action] = winner
        else:
            state.Qa[action] = (state.Na[action] * state.Qa[action] + winner) / (state.Na[action] + 1)

        state.N += 1
        state.Na[action] += 1

    def __select_action(self, state: MCTS_State, canonical: np.array):        
        nSqrt = math.sqrt(state.N)
        def uct(action: int):
            exploitation = state.Qa[action]
            exploration = state.PI[action] * nSqrt / (1 + state.Na[action])
            return exploitation + exploration
        return sorted(self.game.get_valid_actions(canonical), key=lambda x: uct(x), reverse=True)[0]

