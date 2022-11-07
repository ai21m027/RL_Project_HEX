from __future__ import print_function
import sys
sys.path.append('..')

from Game import Game
import numpy as np

from hex.hex_engine import hexPosition as Board

class HexGame(Game):

    def boardCloneInput(self, board):
        board = np.copy(board)

        for i in range(self.n):
            for j in range(self.n):
                if board[i][j] == 2:
                    board[i][j] = -1

        return board

    def boardCloneOutput(self, board):
        board = np.copy(board)

        for i in range(self.n):
            for j in range(self.n):
                if board[i][j] == -1:
                    board[i][j] = 2
                    
        return board

        


    def __init__(self, size):
        self.n = size

    def getInitBoard(self):
        # return initial board (numpy board)
        b = Board(self.n)
        return np.array(b.board)

    def getBoardSize(self):
        # (a,b) tuple
        return (self.n, self.n)

    def getActionSize(self):
        # return number of actions
        return self.n * self.n

    def getNextState(self, board, player, action):

        copy = np.copy(board)

        x, y = int(action / self.n), action % self.n

        copy[x][y] = player

        return (copy, -player)

    def getValidMoves(self, board, player):

        valids = [0] * self.getActionSize()

        b = Board(self.n)
        b.board = self.boardCloneOutput(board)
        legalMoves = b.getActionSpace()
        
        if len(legalMoves)==0:
            return np.array(valids)

        for x, y in legalMoves:
            valids[self.n * x + y] = 1
        
        return np.array(valids)

    def getGameEnded(self, board, player):

        b = Board(self.n)
        b.board = self.boardCloneOutput(board)

        if b.whiteWin():
            return 1 if player == 1 else -1
        if b.blackWin():
            return 1 if player == -1 else -1 
        else:
            return 0

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        # return player * board
        
        if player == 1:
            return board
        else:
            return np.fliplr(np.rot90(-1*board, axes=(1, 0)))

    def getSymmetries(self, board, pi):
        # mirror, rotational
        assert(len(pi) == self.n**2)  # 1 for pass
        pi_board = np.reshape(pi, (self.n, self.n))
        l = []

        for i in [0, 2]:
            newB = np.rot90(board, i)
            newPi = np.rot90(pi_board, i)
            l += [(newB, list(newPi.ravel()))]
        return l

    def stringRepresentation(self, board):

        return board.tostring()