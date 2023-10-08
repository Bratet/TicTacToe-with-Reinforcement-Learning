from minimax import TicTacToeMinimax
from mdp import TicTacToeMDP
from mc import MontoCarlo

if __name__ == "__main__":
    # game = TicTacToeMDP()
    # game = TicTacToeMinimax()
    game = MontoCarlo()
    game.play()
