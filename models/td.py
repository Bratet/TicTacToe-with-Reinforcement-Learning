from tqdm import tqdm
from .utils import TicTacToe
import random
import numpy as np
import pickle
from os.path import exists


class TemporalDifference(TicTacToe):
    OPPONENT_POLICY_PATH = 'models/policies/mdp_policy.pkl'
    POLICY_PATH = 'models/policies/td_policy.pkl'

    def __init__(self, epsilon=0.2, alpha=0.1, gamma=0.9):
        super().__init__()
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.target_policy = {}
        self.behavior_policy = self.epsilon_greedy_policy
        self.opponent_policy = self.load_opponent_policy()
        self.Q = {self.PLAYER_X: {}, self.PLAYER_O: {}}
        self.current_player = self.PLAYER_X

        if exists(self.POLICY_PATH):
            self.load_policy()
        else:
            self.train()
            self.save_policy()

    def load_opponent_policy(self):
        with open(self.OPPONENT_POLICY_PATH, 'rb') as f:
            return pickle.load(f)

    def opponent_move(self, game):
        state = tuple(map(tuple, game.board))
        if state in self.opponent_policy:
            return self.opponent_policy[state]
        else:
            return random.choice(self.possible_actions(state))

    def load_policy(self):
        with open(self.POLICY_PATH, 'rb') as f:
            self.policy = pickle.load(f)

    def save_policy(self):
        with open(self.POLICY_PATH, 'wb') as f:
            pickle.dump(self.policy, f)

    def epsilon_greedy_policy(self, state, Q):
        possible_acts = self.possible_actions(state)
        num_actions = len(possible_acts)
        probs = [self.epsilon / num_actions for _ in range(num_actions)]

        if state in Q:
            q_values = [Q[state].get(action, 0) for action in possible_acts]
            if sum(q_values) != 0:
                max_val = max(q_values)
                best_actions = [idx for idx, q in enumerate(q_values) if q == max_val]
                for idx in best_actions:
                    probs[idx] += (1.0 - self.epsilon) / len(best_actions)

        total = sum(probs)
        probs = [p / total for p in probs]

        return probs, possible_acts

    def simulate_result(self, game, action):
        new_game = TicTacToe()
        new_game.board = np.copy(game.board)
        new_game.board[action[0]][action[1]] = (
            self.PLAYER_X if not game.x_is_human else self.PLAYER_O
        )
        return new_game

    def update_Q(self, state, action, reward, next_state):
        current_estimate = self.Q[self.current_player].get(state, {}).get(action, 0)

        max_next_Q = max(
            self.Q[self.current_player].get(next_state, {}).values(), default=0
        )
        td_target = reward + self.gamma * max_next_Q

        td_error = td_target - current_estimate

        if state not in self.Q[self.current_player]:
            self.Q[self.current_player][state] = {}

        self.Q[self.current_player][state][action] = (
            current_estimate + self.alpha * td_error
        )

    def generate_episode(self):
        current_game = TicTacToe()
        while not current_game.terminal(current_game.board):
            state = tuple(map(tuple, current_game.board))
            current_player = (
                self.PLAYER_X
                if (sum(sum(current_game.board)) % 2 == 0)
                else self.PLAYER_O
            )

            if current_player == self.current_player:
                action_probs, possible_acts = self.epsilon_greedy_policy(
                    state, self.Q[current_player]
                )
                chosen_index = np.random.choice(len(possible_acts), p=action_probs)
                action = possible_acts[chosen_index]
            else:
                action = self.opponent_move(current_game)

            next_game = self.simulate_result(current_game, action)

            if next_game.terminal(next_game.board):
                winner = next_game.winner(next_game.board)
                if winner == self.PLAYER_X:
                    reward = 1 if self.current_player == self.PLAYER_X else -1
                elif winner == self.PLAYER_O:
                    reward = -1 if self.current_player == self.PLAYER_O else 1
                else:
                    reward = 0
            else:
                reward = 0

            next_state = tuple(map(tuple, next_game.board))
            self.update_Q(state, action, reward, next_state)
            current_game = next_game

    def train(self, num_episodes=1_00_000, split_ratio=0.5):
        x_episodes = int(split_ratio * num_episodes)
        o_episodes = num_episodes - x_episodes

        for player, episodes in [
            (self.PLAYER_X, x_episodes),
            (self.PLAYER_O, o_episodes),
        ]:
            self.current_player = player
            for _ in tqdm(range(episodes)):
                self.generate_episode()

        self.policy = {}
        for player in [self.PLAYER_X, self.PLAYER_O]:
            self.policy.update(
                {
                    state: max(self.Q[player][state], key=self.Q[player][state].get)
                    for state in self.Q[player]
                }
            )

    def best_action(self):
        state = tuple(map(tuple, self.board))
        if state in self.policy:
            return self.policy[state]
        else:
            return random.choice(self.possible_actions(state))
