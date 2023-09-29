from utils import TicTacToe
import numpy as np

class TicTacToeMinimax(TicTacToe):
    
    def __init__(self):
        super().__init__()
    
    def minimax(self, board, alpha, beta, maximizing):
        if self.x_is_human:
            if board.terminal():
                return board.utility(2)

            if maximizing:
                value = -np.inf
                for action in board.actions():
                    value = max(
                        value, self.minimax(board.result(action), alpha, beta, False)
                    )
                    alpha = max(alpha, value)
                    if beta <= alpha:
                        break
                return value
            else:
                value = np.inf
                for action in board.actions():
                    value = min(
                        value, self.minimax(board.result(action), alpha, beta, True)
                    )
                    beta = min(beta, value)
                    if beta <= alpha:
                        break
                return value
        else:
            if board.terminal():
                return board.utility(1)

            if maximizing:
                value = -np.inf
                for action in board.actions():
                    value = max(
                        value, self.minimax(board.result(action), alpha, beta, False)
                    )
                    alpha = max(alpha, value)
                    if beta <= alpha:
                        break
                return value
            else:
                value = np.inf
                for action in board.actions():
                    value = min(
                        value, self.minimax(board.result(action), alpha, beta, True)
                    )
                    beta = min(beta, value)
                    if beta <= alpha:
                        break
                return value
    
    def best_action(self):
        value = -np.inf
        alpha = -np.inf
        beta = np.inf
        best_action = None
        for action in self.actions():
            move_value = self.minimax(self.result(action), alpha, beta, False)
            if move_value > value:
                value = move_value
                best_action = action
        return best_action