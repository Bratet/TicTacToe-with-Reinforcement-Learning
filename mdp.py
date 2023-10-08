import numpy as np
import tkinter as tk
from tkinter import messagebox
import os
import random
import json
from utils import TicTacToe
from tqdm import tqdm

class TicTacToeMDP(TicTacToe):
    iterations = 100

    def __init__(self):
 
        if not os.path.isfile("policies/policy_X.json") or not os.path.isfile("policies/policy_O.json"):
            if not os.path.isdir("policies"):
                os.mkdir("policies")
            
            print(f"Training Value Iteration for the X player for {self.iterations} iterations...")
            self.policy_X = self.compute_value_iteration(self.PLAYER_X)
            print(f"Training Value Iteration for the O player for {self.iterations} iterations...")
            self.policy_O = self.compute_value_iteration(self.PLAYER_O)

            self.board = np.zeros((3, 3))

            with open("policies/policy_X.json", "w") as f:
                json.dump(self.policy_X, f)

            with open("policies/policy_O.json", "w") as f:
                json.dump(self.policy_O, f)
        else:
            with open("policies/policy_X.json", "r") as f:
                self.policy_X = json.load(f)

            with open("policies/policy_O.json", "r") as f:
                self.policy_O = json.load(f)

    def is_valid_state(self, state):
        count_x = sum([1 for s in state if s == '1'])
        count_o = sum([1 for s in state if s == '2'])
        return abs(count_x - count_o) <= 1

    def compute_value_iteration(self, player, gamma=1):
        V = {}

        for _ in tqdm(range(self.iterations)):
            for i in range(3**9):
                state = np.base_repr(i, base=3).zfill(9)
                if not self.is_valid_state(state):
                    continue
                self.board = np.array(list(map(int, list(state)))).reshape(3, 3)

                if self.terminal():
                    V[state] = self.utility(player)
                    continue

                max_value = float('-inf')
                for action in self.actions():
                    temp_board = self.result(action).board
                    next_state = ''.join(map(str, temp_board.flatten()))
                    next_value = self.utility(player) + gamma * V.get(next_state, 0)
                    max_value = max(max_value, next_value)

                V[state] = max_value

        policy = {}
        for state, value in V.items():
            self.board = np.array(list(map(int, list(state)))).reshape(3, 3)
            best_action = None
            best_value = float('-inf')
            for action in self.actions():
                temp_board = self.result(action).board
                r = self.utility(player)
                next_state = ''.join(map(str, temp_board.flatten()))
                next_value = r + gamma * V.get(next_state, 0)
                if next_value > best_value:
                    best_value = next_value
                    best_action = action

            policy[state] = best_action

        return policy

    def best_action(self):
        state = self.board.astype(int).flatten()
        state = ''.join(map(str, state))
        if self.player() == self.PLAYER_X:
            return tuple(self.policy_X[state])
        else:
            return tuple(self.policy_O[state])
