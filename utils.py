import numpy as np
import tkinter as tk
from tkinter import messagebox
import os
from tqdm import tqdm

class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3,3))
        self.game_over = False
        self.values = {}
        self.alpha = 0
        
    def create_board(self):
        root = tk.Tk()
        root.title("Tic Tac Toe")
        root.resizable(0, 0)
        root.configure(background='black')
        
        self.buttons = []
        for i in range(3):
            self.buttons.append([])
            for j in range(3):
                self.buttons[i].append(tk.Button(root, width=10, height=5, command=lambda row=i, col=j: self.click(row, col)))
                self.buttons[i][j].grid(row=i, column=j)
                
        root.mainloop()
        
    def click(self, row, col):
        if self.game_over == False:
            self.buttons[row][col].configure(text="X", state="disabled")
            self.board[row][col] = 1
            if self.terminal():
                self.show_winner()
                self.game_over = True
                return
            self.computer_move()
            if self.terminal():
                self.show_winner()
                self.game_over = True
                return
            
    def computer_move(self):
        action = self.best_action()
        if action:
            self.buttons[action[0]][action[1]].configure(text="O", state="disabled")
            self.board[action[0]][action[1]] = 2
        
    def player(self):
        num_ones = np.count_nonzero(self.board == 1)
        num_twos = np.count_nonzero(self.board == 2)

        if num_ones == num_twos:
            return 1
        else:
            return 2
    
    def actions(self):
        actions = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == 0:
                    actions.append((i,j))
        return actions
    
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
    
    def get_all_states(self):
        all_states = set()
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        for m in range(3):
                            for n in range(3):
                                for o in range(3):
                                    for p in range(3):
                                        for q in range(3):
                                            all_states.add((i,j,k,l,m,n,o,p,q))
                                            
        for state in list(all_states):
            board = np.array(state).reshape(3,3)
            num_ones = np.count_nonzero(board == 1)
            num_twos = np.count_nonzero(board == 2)
            if abs(num_ones - num_twos) > 1:
                all_states.remove(state)

        all_states = [np.array(state).reshape(3,3) for state in list(all_states)]
                
        return all_states
    
    def train(self, iterations=100, alpha=0.9):
        states = self.get_all_states()
        self.alpha = alpha
        self.values = {}

            
            
        for state in states:
            self.values[tuple(state.flatten())] = 0
            
        for _ in tqdm(range(iterations)):
            delta = 0
            new_values = self.values.copy()
                
            for state in states:
                board = TicTacToe()
                board.board = state
                
                if board.terminal():
                    new_values[tuple(state.flatten())] = board.utility(board.player())
                    continue

                if board.player() == 1:
                    max_value = -np.inf
                    for action in board.actions():
                        next_state = board.result(action)
                        value = next_state.utility(1) + self.alpha * self.values.get(tuple(next_state.board.flatten()), 0)
                        max_value = max(max_value, value)
                    new_values[tuple(state.flatten())] = max_value
                else:
                    min_value = np.inf
                    for action in board.actions():
                        next_state = board.result(action)
                        value = next_state.utility(2) + self.alpha * self.values.get(tuple(next_state.board.flatten()), 0)
                        min_value = min(min_value, value)
                    new_values[tuple(state.flatten())] = min_value
                
                delta = max(delta, abs(new_values[tuple(state.flatten())] - self.values[tuple(state.flatten())]))
                
            self.values = new_values


    def utility(self, player):
        if self.winner() == player:
            return 1
        elif self.winner() == 3 - player:
            return -1
        else:
            return 0
        
    def best_action(self):
        max_value = -np.inf
        best_act = None

        for action in self.actions():
            next_state = self.result(action)
            value = self.values.get(tuple(next_state.board.flatten()), 0)
            
            if value > max_value:
                max_value = value
                best_act = action
                    
        return best_act
        
    def show_winner(self):
        winner = self.winner()
        if winner == 1:
            messagebox.showinfo("Game Over", "Player X wins!")
        elif winner == 2:
            messagebox.showinfo("Game Over", "Player O wins!")
        else:
            messagebox.showinfo("Game Over", "It's a draw!")
            
        os._exit(0)
    
    def play(self):
        self.create_board()

# import numpy as np
# import tkinter as tk
# from tkinter import messagebox
# import os

# class TicTacToe:
#     def __init__(self):
#         self.board = np.zeros((3,3))
#         self.game_over = False

        
#     def create_board(self):
#         root = tk.Tk()
#         root.title("Tic Tac Toe")
#         root.resizable(0, 0)
#         root.configure(background='black')
        
#         self.buttons = []
#         for i in range(3):
#             self.buttons.append([])
#             for j in range(3):
#                 self.buttons[i].append(tk.Button(root, width=10, height=5, command=lambda row=i, col=j: self.click(row, col)))
#                 self.buttons[i][j].grid(row=i, column=j)
                
#         root.mainloop()
        
#     def click(self, row, col):
#         if self.game_over == False:
#             self.buttons[row][col].configure(text="X", state="disabled")
#             self.board[row][col] = 1
#             if self.terminal():
#                 self.show_winner()
#                 self.game_over = True
#                 return
#             self.computer_move()
#             if self.terminal():
#                 self.show_winner()
#                 self.game_over = True
#                 return
            
#     def computer_move(self):
#         action = self.best_action()
#         if action:
#             self.buttons[action[0]][action[1]].configure(text="O", state="disabled")
#             self.board[action[0]][action[1]] = 2
        
#     def player(self):
#         num_ones = np.count_nonzero(self.board == 1)
#         num_twos = np.count_nonzero(self.board == 2)

#         if num_ones == num_twos:
#             return 1
#         else:
#             return 2
    
#     def actions(self):
#         actions = []
#         for i in range(3):
#             for j in range(3):
#                 if self.board[i][j] == 0:
#                     actions.append((i,j))
#         return actions
    
#     def result(self, action):
#         new_board = TicTacToe()
#         new_board.board = np.copy(self.board)
#         new_board.board[action[0]][action[1]] = self.player()
#         return new_board
        
#     def winner(self):
#         for i in range(3):
#             if self.board[i][0] == self.board[i][1] == self.board[i][2] != 0:
#                 return self.board[i][0]
#             if self.board[0][i] == self.board[1][i] == self.board[2][i] != 0:
#                 return self.board[0][i]
#         if self.board[0][0] == self.board[1][1] == self.board[2][2] != 0:
#             return self.board[0][0]
#         if self.board[0][2] == self.board[1][1] == self.board[2][0] != 0:
#             return self.board[0][2]
#         return None
        
#     def terminal(self):
#         return self.winner() != None or len(self.actions()) == 0

#     def minimax(self, board, alpha, beta, maximizing):
#         if board.terminal():
#             return board.utility(2)
        
#         if maximizing:
#             value = -np.inf
#             for action in board.actions():
#                 value = max(value, self.minimax(board.result(action), alpha, beta, False))
#                 alpha = max(alpha, value)
#                 if beta <= alpha:
#                     break
#             return value
#         else:
#             value = np.inf
#             for action in board.actions():
#                 value = min(value, self.minimax(board.result(action), alpha, beta, True))
#                 beta = min(beta, value)
#                 if beta <= alpha:
#                     break
#             return value
        
#     def utility(self, player):
#         if self.winner() == player:
#             return 1
#         elif self.winner() == 3 - player:
#             return -1
#         else:
#             return 0
        
#     def best_action(self):
#         value = -np.inf
#         alpha = -np.inf
#         beta = np.inf
#         best_action = None
#         for action in self.actions():
#             move_value = self.minimax(self.result(action), alpha, beta, False)
#             if move_value > value:
#                 value = move_value
#                 best_action = action
#         return best_action

        
#     def show_winner(self):
#         winner = self.winner()
#         if winner == 1:
#             messagebox.showinfo("Game Over", "Player X wins!")
#         elif winner == 2:
#             messagebox.showinfo("Game Over", "Player O wins!")
#         else:
#             messagebox.showinfo("Game Over", "It's a draw!")
            
#         os._exit(0)
    
#     def play(self):
#         self.create_board()
