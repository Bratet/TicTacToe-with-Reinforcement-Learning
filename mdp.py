import json
import random
import tqdm
import numpy as np
from utils import TicTacToe
import os

class TicTacToeMDP(TicTacToe):
    
    def __init__(self):
        super().__init__()
        if not os.path.isfile("policies/policy_X.json") or not os.path.isfile("policies/policy_O.json"):
            if not os.path.isdir("policies"):
                os.mkdir("policies")
            
            print("Training MDP for the X player...")
            self.policy_X = self.compute_policy(1, iterations=5000)
            print("Training MDP for the O player...")
            self.policy_O = self.compute_policy(2, iterations=5000)
            
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

    def initialize_policy(self):
        policy = {}
        center = (1, 1)
        corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
        for i in range(3**9):
            state = np.base_repr(i, base=3).zfill(9)
            if not self.is_valid_state(state):
                continue
            self.board = np.array(list(map(int, list(state)))).reshape(3, 3)
            
            if self.terminal():
                continue
            
            actions = self.actions()
            if actions:
                if center in actions:
                    policy[state] = center
                elif set(corners).intersection(actions):
                    policy[state] = random.choice(list(set(corners).intersection(actions)))
                else:
                    policy[state] = random.choice(actions)
        return policy

    def compute_policy(self, player, iterations=1000):
        policy = self.initialize_policy()
        V = {}
        gamma = 1

        for _ in tqdm.tqdm(range(iterations)):
            V = self.policy_evaluation(policy, V, player, gamma)
            stable = self.policy_improvement(V, policy, player, gamma)
            if stable:
                break
        
        return policy

    def policy_evaluation(self, policy, V, player, gamma=0.9, theta=1e-8):
        while True:
            delta = 0
            for state, action in policy.items():
                old_value = V.get(state, 0)
                self.board = np.array(list(map(int, list(state)))).reshape(3, 3)
                
                if not self.terminal():
                    self.perform_move(action[0], action[1], player)
                    r = self.utility(player)
                    next_state = ''.join(map(str, self.board.flatten()))
                    next_value = V.get(next_state, 0)
                    V[state] = r + gamma * next_value
                    delta = max(delta, abs(V[state] - old_value))
            if delta < theta:
                break
        return V

    def policy_improvement(self, V, policy, player, gamma=1):
        stable = True
        new_policy = {}
        for state in V:
            self.board = np.array(list(map(int, list(state)))).reshape(3, 3)
            actions = self.actions()

            best_action = None
            best_value = float('-inf')
            for action in actions:
                temp_board = self.board.copy()
                temp_board[action[0], action[1]] = player
                r = self.utility(player)
                next_state = ''.join(map(str, temp_board.flatten()))
                next_value = r + gamma * V.get(next_state, 0)
                if next_value > best_value:
                    best_value = next_value
                    best_action = action

            new_policy[state] = best_action
            if policy[state] != best_action:
                stable = False
        return stable
    
    def perform_move(self, row, col, player):
        if self.board[row][col] == 0:
            self.board[row][col] = player

    def best_action(self):
        state = self.board.astype(int).flatten()
        state = list(state)
        state = ''.join(map(str, state))
  
        if self.player() == self.PLAYER_X:
            return tuple(self.policy_X[state])
        else:
            return tuple(self.policy_O[state])
