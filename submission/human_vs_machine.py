import pickle
import sys
sys.path.append('..')
from hex_engine import hexPosition


player = pickle.load(open('player.pkl', 'rb'))

target = hexPosition(7)

target.humanVersusMachine(1, lambda board,_: player.select_move(board, 2))