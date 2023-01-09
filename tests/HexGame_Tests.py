import unittest
import numpy as np
from HexGame import HexGame


Player1 = 1
Player2 = -1

X = Player1
O = Player2
_ = 0


class HexGameTests(unittest.TestCase):

    def setUp(self):
        self.game = HexGame(3)

    def test_get_winner(self):

        self.__assert_winner([
            _, _, _,
            _, _, _,
            _, _, _
        ], winner=0)

        self.__assert_winner([
            X, X, X,
            _, _, _,
            _, _, _
        ], winner=Player1)

        self.__assert_winner([
            O, O, O,
            _, _, _,
            _, _, _
        ], winner=0)

        self.__assert_winner([
            X, _, _,
            X, _, _,
            X, _, _
        ], winner=0)

        self.__assert_winner([
            O, _, _,
            O, _, _,
            O, _, _
        ], winner=Player2)

        self.__assert_winner([
            _, _, X,
            _, X, _,
            X, _, _
        ], winner=Player1)

        self.__assert_winner([
            _, _, O,
            _, O, _,
            O, _, _
        ], winner=Player2)

        self.__assert_winner([
            X, _, _,
            _, X, _,
            _, _, X
        ], winner=0)

        self.__assert_winner([
            O, _, _,
            _, O, _,
            _, _, O
        ], winner=0)

    def test_get_symmetries(self):

        board = self.__transform_board([
            O, _, O,
            O, O, X,
            O, X, _
        ])
        pi = self.__transform_board([
            0, 1, 0,
            0, 0, 0,
            0, 0, 1
        ]).ravel()

        expected_boards = [
            self.__transform_board([
                O, _, O,
                O, O, X,
                O, X, _
            ]),
            self.__transform_board([
                _, X, O,
                X, O, O,
                O, _, O
            ])
        ]

        expected_pis = [
            self.__transform_board([
                0, 1, 0,
                0, 0, 0,
                0, 0, 1
            ]).ravel(),
            self.__transform_board([
                1, 0, 0,
                0, 0, 0,
                0, 1, 0
            ]).ravel()
        ]

        boards, pis = self.game.get_symmetries(board, pi)

        for i in range(len(boards)):
            self.assertTrue(np.array_equal(boards[i], expected_boards[i]))
            self.assertTrue(np.array_equal(pis[i], expected_pis[i]))

    def test_make_move_xy(self):
        board = self.game.get_init_board()
        self.game.make_move_xy(board, 0, 0, 1)
        self.assertEqual(board[0, 0], 1)

        self.game.make_move_xy(board, 2, 0, 1)
        self.assertEqual(board[0, 2], 1)

    def test_make_move(self):
        board = self.game.get_init_board()
        self.game.make_move(board, 0, 1)
        self.assertEqual(board[0, 0], 1)

        self.game.make_move(board, 2, 1)
        self.assertEqual(board[0, 2], 1)

    def test_make_moves(self):
        board = self.game.get_init_board()
        
        for _ in range(self.game.action_size):
            actions = list(self.game.get_valid_actions(board))
            action = actions[0]
            self.game.make_move(board, action, 1)

    def test_make_moves2(self):
        board = self.game.get_init_board()
        
        for _ in range(self.game.action_size):
            canonical = self.game.get_canonical_board(board, -1)
            action = list(self.game.get_valid_actions(canonical))[0]
            x, y = self.game.get_original_action(action, -1)
            self.game.make_move_xy(board, x, y, -1)

    def test_get_original_action(self):
        x, y = self.game.get_original_action(0, 1)
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

        x, y = self.game.get_original_action(0, -1)
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

        x, y = self.game.get_original_action(2, 1)
        self.assertEqual(x, 2)
        self.assertEqual(y, 0)

        x, y = self.game.get_original_action(2, -1)
        self.assertEqual(x, 0)
        self.assertEqual(y, 2)



    def __assert_winner(self, board: list, winner: int):
        board = self.__transform_board(board)

        self.assertEqual(self.game.has_player1_won(board), winner == 1)
        self.assertEqual(self.game.has_player2_won(board), winner == -1)
        pass

    def __transform_board(self, board: list):
        return np.array(board).reshape((3, 3))
