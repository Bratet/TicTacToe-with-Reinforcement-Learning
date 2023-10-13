from tqdm import tqdm
from .utils import TicTacToe
import random
import numpy as np
import pickle
from os.path import exists


class MonteCarlo(TicTacToe):
    OPPONENT_POLICY_PATH = 'models/policies/mdp_policy.pkl'

    POLICY_PATH = 'models/policies/mc_policy.pkl'

    def __init__(self, epsilon=0.2):
        super().__init__()
        self.epsilon = epsilon
        self.target_policy = {}
        self.behavior_policy = self.epsilon_greedy_policy
        self.opponent_policy = self.load_opponent_policy()
        self.policy = {}
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

    def generate_episode(self, Q):
        current_game = TicTacToe()
        states, actions, rewards, probs = [], [], [], []

        while not current_game.terminal(current_game.board):
            state = tuple(map(tuple, current_game.board))

            current_player = (
                self.PLAYER_X
                if (sum(sum(current_game.board)) % 2 == 0)
                else self.PLAYER_O
            )

            if current_player == self.current_player:
                action_probs, possible_acts = self.epsilon_greedy_policy(state, Q)
                chosen_index = np.random.choice(len(possible_acts), p=action_probs)
                action = possible_acts[chosen_index]
                chosen_prob = action_probs[chosen_index]

            else:
                action = self.opponent_move(current_game)
                possible_acts = self.possible_actions(state)
                action_probs = [1.0 if act == action else 0.0 for act in possible_acts]
                chosen_prob = 1.0 if action in possible_acts else 0.0

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

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            probs.append(chosen_prob)

            current_game = next_game

        return states, actions, rewards, probs

    def train(self, discount_factor=0.9, num_episodes=1_000_000, split_ratio=0.5):
        Q = {self.PLAYER_X: {}, self.PLAYER_O: {}}
        C = {self.PLAYER_X: {}, self.PLAYER_O: {}}

        x_episodes = int(split_ratio * num_episodes)
        o_episodes = num_episodes - x_episodes

        for player, episodes in [
            (self.PLAYER_X, x_episodes),
            (self.PLAYER_O, o_episodes),
        ]:
            self.current_player = player
            for _ in tqdm(range(episodes)):
                states, actions, rewards, behavior_probs = self.generate_episode(
                    Q[player]
                )
                G = 0
                W = 1
                for state, action, reward, prob in zip(
                    reversed(states),
                    reversed(actions),
                    reversed(rewards),
                    reversed(behavior_probs),
                ):
                    G = discount_factor * G + reward

                    if state not in C[player]:
                        C[player][state] = {}
                    if state not in Q[player]:
                        Q[player][state] = {}

                    if action not in C[player][state]:
                        C[player][state][action] = 0
                    if action not in Q[player][state]:
                        Q[player][state][action] = 0

                    C[player][state][action] += W
                    Q[player][state][action] += (W / C[player][state][action]) * (
                        G - Q[player][state][action]
                    )
                    W *= 1.0 / prob

                    self.epsilon = max(0.1, self.epsilon * 0.999999)

        self.policy = {}
        for player in [self.PLAYER_X, self.PLAYER_O]:
            self.policy.update(
                {
                    state: max(Q[player][state], key=Q[player][state].get)
                    for state in Q[player]
                }
            )

    def best_action(self):
        state = tuple(map(tuple, self.board))
        if state in self.policy:
            return self.policy[state]
        else:
            return random.choice(self.possible_actions(state))
