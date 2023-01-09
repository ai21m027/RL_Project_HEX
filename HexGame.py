from collections import deque
import numpy as np


class HexGame:
    Adjacents = [(0, 1), (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1)]

    def __init__(self, size: int) -> None:
        self.size = size
        self.action_size = size * size

    def get_init_board(self) -> np.array:
        return np.zeros((self.size, self.size), dtype=np.int32)

    def get_valid_actions(self, board: np.array):
        i = 0
        for y in range(self.size):
            for x in range(self.size):
                if board[y, x] == 0:
                    yield i
                i += 1

    def get_action_mask(self, board: np.array):
        mask = np.zeros(self.action_size)
        for action in self.get_valid_actions(board):
            mask[action] = 1
        return mask

    def make_move(self, board: np.array, idx: int, player: int):
        self.make_move_xy(board, idx % self.size, idx // self.size, player)

    def make_move_xy(self, board: np.array, x: int, y: int, player: int):
        assert x >= 0 and x < self.size
        assert y >= 0 and y < self.size
        assert player == 1 or player == -1
        assert board[y, x] == 0

        board[y, x] = player

    def get_state_idx(self, board: np.array):
        state = 0
        i = 0
        for y in range(self.size):
            for x in range(self.size):
                value = board[y, x]
                if value == 1:
                    state |= 1 << i
                elif value == -1:
                    state |= 1 << (i + 1)
                i += 2
        return state

    def get_original_action(self, action: int, player: int):
        assert player == 1 or player == -1

        if player == 1:
            return action % self.size, action // self.size
        else:
            x = action % self.size
            y = action // self.size

            board = np.zeros(shape=(self.size, self.size))
            board[y][x] = 1

            board = self.get_original_board(board, -1)
            loc = np.where(board == -1)
            x, y = loc[1][0], loc[0][0]
            return (x, y)

    def get_canonical_board(self, board: np.array, player: int):
        assert player == 1 or player == -1

        if player == 1:
            return board.copy()
        else:
            return np.fliplr(np.rot90(-1 * board, axes=(1, 0)))

    def get_original_board(self, board: np.array, player: int):
        if player == 1:
            return board
        else:
            return np.rot90(np.fliplr(-1 * board), axes=(0, 1))

    def get_symmetries(self, board: np.array, pi: np.array):
        assert len(pi) == self.action_size

        boards = [board, np.rot90(board, 2)]
        pis = [pi, np.rot90(np.reshape(pi, (self.size, self.size)), 2).ravel()]

        return (boards, pis)

    def has_player1_won(self, board: np.array):
        visited = set()
        next = deque()
        for i in range(self.size):
            if board[i, 0] == 1:
                next.append((0, i))

                while len(next) > 0:
                    x, y = next.popleft()
                    if x == self.size - 1:
                        return True

                    if ((x, y) in visited):
                        continue

                    visited.add((x, y))

                    for dx, dy in self.Adjacents:
                        nx = x + dx
                        ny = y + dy

                        if nx < 0 or nx >= self.size or ny < 0 or ny >= self.size:
                            continue

                        if board[ny, nx] == 1:
                            next.append((nx, ny))
        return False

    def has_player2_won(self, board: np.array):
        visited = set()
        next = deque()
        for i in range(self.size):
            if board[0, i] == -1:
                next.append((i, 0))

                while len(next) > 0:
                    x, y = next.popleft()
                    if y == self.size - 1:
                        return True

                    if ((x, y) in visited):
                        continue

                    visited.add((x, y))

                    for dx, dy in self.Adjacents:
                        nx = x + dx
                        ny = y + dy

                        if nx < 0 or nx >= self.size or ny < 0 or ny >= self.size:
                            continue

                        if board[ny, nx] == -1:
                            next.append((nx, ny))
        return False

    def print(self, board: np.array) -> str:

        s = ''
        for y in range(self.size):
            for x in range(self.size):
                if board[y, x] == 1:
                    s += 'X '
                elif board[y, x] == -1:
                    s += 'O '
                else:
                    s += '_ '
            s += '\n'

        print(s)
