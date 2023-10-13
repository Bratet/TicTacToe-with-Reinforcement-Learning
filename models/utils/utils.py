import numpy as np
import tkinter as tk
from tkinter import messagebox
import os
import random


class TicTacToe:
    PLAYER_X = 1
    PLAYER_O = 2

    def __init__(self, state=None):
        if state:
            self.board = np.array(state)
        else:
            self.board = np.zeros((3, 3))
        self.game_over = False
        self.x_is_human = random.choice([True, False])

    def create_board(self):
        root = tk.Tk()
        root.title("Tic Tac Toe")
        root.resizable(0, 0)
        root.configure(background='white')

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

        self.reset_button = tk.Button(root, text="Reset", command=self.reset_game_gui)
        self.reset_button.grid(row=3, column=0, columnspan=3)

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
            if self.terminal(self.board):
                self.show_winner()
                self.game_over = True

    def reset_game_gui(self):
        self.reset_game()
        for i in range(3):
            for j in range(3):
                self.buttons[i][j].configure(text="", state="normal")
        if not self.x_is_human:
            self.computer_move()

    def computer_move(self):
        action = self.best_action()
        if action:
            self.make_move(
                action[0],
                action[1],
                self.PLAYER_X if not self.x_is_human else self.PLAYER_O,
            )

    def possible_actions(self, board):
        return [(i, j) for i in range(3) for j in range(3) if board[i][j] == 0]

    def winner(self, board):
        for i in range(3):
            if board[i][0] == board[i][1] == board[i][2] != 0:
                return board[i][0]
            if board[0][i] == board[1][i] == board[2][i] != 0:
                return board[0][i]
        if board[0][0] == board[1][1] == board[2][2] != 0:
            return board[0][0]
        if board[0][2] == board[1][1] == board[2][0] != 0:
            return board[0][2]
        return None

    def terminal(self, board):
        return self.winner(board) != None or len(self.possible_actions(board)) == 0

    def best_action(self):
        pass

    def show_winner(self):
        winner = self.winner(self.board)
        if winner == self.PLAYER_X:
            messagebox.showinfo("Game Over", "Player X wins!")
        elif winner == self.PLAYER_O:
            messagebox.showinfo("Game Over", "Player O wins!")
        else:
            messagebox.showinfo("Game Over", "It's a draw!")

    def play(self):
        self.board = np.zeros((3, 3))
        self.create_board()

    def reset_game(self):
        self.board = np.zeros((3, 3))
        self.game_over = False
        self.x_is_human = random.choice([True, False])

    def turn(self):
        empty_spaces = np.sum(self.board == 0)
        return self.PLAYER_X if empty_spaces % 2 == 1 else self.PLAYER_O
