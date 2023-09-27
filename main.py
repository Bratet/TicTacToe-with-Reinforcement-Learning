from utils import TicTacToe

if __name__ == "__main__":
    game = TicTacToe()
    game.train(100, alpha=0.9)
    # import pdb; pdb.set_trace()
    game.play()