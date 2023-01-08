import math
import numpy as np
import sys
sys.path.append('hex/')
from HexGame import display

EPS = 1e-8

class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.numMCTSSims = args.numMCTSSims
        self.cpuct = args.cpuct
        self.states = {}
        self.sim_count = 0

    def getActionProb(self, board, player, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """

        canonicalBoard = self.game.getCanonicalForm(board, player)

        for _ in range(self.numMCTSSims):
            self.search(board, player)

        s = self.game.stringRepresentation(canonicalBoard)
        state = self.states[s]

        counts = [state.Na[a] if a in state.Na else 0 for a in range(self.game.getActionSize())]

        if temp == 0:
            bestA = np.argmax(counts)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x**(1. / temp) for x in counts]
        probs = [x / float(sum(counts)) for x in counts]
        return probs


    def search(self, board, player):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propogated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propogated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        self.sim_count += 1

        canonicalBoard = self.game.getCanonicalForm(board, player)

        s = self.game.stringRepresentation(canonicalBoard)
        state = None

        if s not in self.states:
            state = State()
            state.ended = self.game.getGameEnded(canonicalBoard, 1)
            state.valids = self.game.getValidMoves(canonicalBoard, 1)
            self.states[s] = state
        else:
            state = self.states[s]
        
        if state.ended != 0:
            # terminal node
            return -state.ended

        if state.preds is None:
            # leaf node
            state.preds, v = self.nnet.predict(canonicalBoard)
            state.preds = state.preds * state.valids      # masking invalid moves
            sum_Ps_s = np.sum(state.preds)

            if sum_Ps_s > 0:
                state.preds /= sum_Ps_s    # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable
                
                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                print("All valid moves were masked, do workaround.")
                state.preds = state.preds + state.valids
                state.preds /= np.sum(state.preds)

            return -v

        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()):
            if state.valids[a]:
                if a in state.Qa:
                    u = state.Qa[a] + self.cpuct * state.preds[a] * math.sqrt(state.N) / (1 + state.Na[a])
                else:
                    u = self.cpuct * state.preds[a] * math.sqrt(state.N + EPS)     # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act

        if state.valids[a] == 0:
            print('invalid action in MCTS', a)
            assert state.valids[a] > 0         

        next_s, _ = self.game.getNextState(canonicalBoard, 1, a)        
        next_s = self.game.getOriginalForm(next_s, player)
        next_player = -player

        v = self.search(next_s, next_player)

        if a in state.Qa:
            state.Qa[a] = (state.Na[a] * state.Qa[a] + v) / (state.Na[a] + 1)
            state.Na[a] += 1

        else:
            state.Qa[a] = v
            state.Na[a] = 1

        state.N += 1
        return -v

class State():
    def __init__(self):
        self.ended = 0
        self.preds = None
        self.valids = None
        self.N = 0
        self.Qa = {}
        self.Na = {}