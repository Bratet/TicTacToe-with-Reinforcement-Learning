import numpy as np
import tkinter as tk
from tkinter import messagebox
import os
import random


class TicTacToe:
    PLAYER_X = 1
    PLAYER_O = 2

    def __init__(self):
        self.board = np.zeros((3, 3))
        self.game_over = False
        self.x_is_human = random.choice([True, False])

    def play(self):
        self.create_board()

    def create_board(self):
        root = tk.Tk()
        root.title("Tic Tac Toe")
        root.resizable(0, 0)
        root.configure(background='black')

        self.buttons = [
            [
                tk.Button(
                    root,
                    width=10,
                    height=5,
                    command=lambda row=i, col=j: self.click(row, col),
                )
                for j in range(3)
            ]
            for i in range(3)
        ]

        for i in range(3):
            for j in range(3):
                self.buttons[i][j].grid(row=i, column=j)

        if not self.x_is_human:
            self.computer_move()

        root.mainloop()

    def click(self, row, col):
        if self.game_over:
            return

        if self.x_is_human:
            self.make_move(row, col, self.PLAYER_X)
            if not self.game_over:
                self.computer_move()
        else:
            self.make_move(row, col, self.PLAYER_O)
            if not self.game_over:
                self.computer_move()

    def make_move(self, row, col, player):
        if self.board[row][col] == 0:
            self.board[row][col] = player
            self.buttons[row][col].configure(
                text=("X" if player == self.PLAYER_X else "O"), state="disabled"
            )
            if self.terminal():
                self.show_winner()
                self.game_over = True

    def computer_move(self):
        action = self.best_action()
        if action:
            self.make_move(
                action[0],
                action[1],
                self.PLAYER_X if not self.x_is_human else self.PLAYER_O,
            )

    def player(self):
        return (
            self.PLAYER_X
            if np.count_nonzero(self.board == self.PLAYER_X)
            == np.count_nonzero(self.board == self.PLAYER_O)
            else self.PLAYER_O
        )

    def actions(self):
        return [(i, j) for i in range(3) for j in range(3) if self.board[i][j] == 0]

    def result(self, action):
        new_board = TicTacToe()
        new_board.board = np.copy(self.board)
        new_board.board[action[0]][action[1]] = self.player()
        return new_board

    def winner(self):
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] != 0:
                return self.board[i][0]
            if self.board[0][i] == self.board[1][i] == self.board[2][i] != 0:
                return self.board[0][i]
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != 0:
            return self.board[0][0]
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != 0:
            return self.board[0][2]
        return None

    def terminal(self):
        return self.winner() != None or len(self.actions()) == 0

    def utility(self, player):
        if self.winner() == player:
            return 1
        elif self.winner() == 3 - player:
            return -1
        else:
            return 0

    def best_action(self):
        pass

    def show_winner(self):
        winner = self.winner()
        if winner == self.PLAYER_X:
            messagebox.showinfo("Game Over", "Player X wins!")
        elif winner == self.PLAYER_O:
            messagebox.showinfo("Game Over", "Player O wins!")
        else:
            messagebox.showinfo("Game Over", "It's a draw!")

        os._exit(0)

    def play(self):
        self.create_board()
