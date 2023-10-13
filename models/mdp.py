from .utils import TicTacToe
import numpy as np
from tqdm import tqdm
import pickle
from os.path import exists


class MDP(TicTacToe):
    POLICY_FILE = 'models/policies/mdp_policy.pkl'

    def __init__(self):
        super().__init__()
        self.states = self.generate_all_states()
        self.V = {}
        for state in self.states:
            if self.terminal(state):
                if self.winner(state) == self.PLAYER_X:
                    self.V[tuple(map(tuple, state))] = 1
                elif self.winner(state) == self.PLAYER_O:
                    self.V[tuple(map(tuple, state))] = -1
                else:
                    self.V[tuple(map(tuple, state))] = 0
            else:
                self.V[tuple(map(tuple, state))] = 0
        self.policy = {}

        if exists(self.POLICY_FILE):
            self.load_policy()
        else:
            self.train()
            self.save_policy()

    def result(self, state, action, player):
        new_state = state.copy()
        new_state[action[0]][action[1]] = player
        return new_state

    def reward(self, state, player):
        winner = self.winner(state)
        if winner == player:
            return 1
        elif winner is not None:
            return -1
        else:
            # Intermediate rewards/punishments
            for i in range(3):
                if (
                    np.count_nonzero(state[i, :] == player) == 2
                    and np.count_nonzero(state[i, :] == 0) == 1
                ):
                    return 0.5
                if (
                    np.count_nonzero(state[:, i] == player) == 2
                    and np.count_nonzero(state[:, i] == 0) == 1
                ):
                    return 0.5
                if (
                    np.count_nonzero(np.diag(state) == player) == 2
                    and np.count_nonzero(np.diag(state) == 0) == 1
                ):
                    return 0.5
                if (
                    np.count_nonzero(np.diag(np.fliplr(state)) == player) == 2
                    and np.count_nonzero(np.diag(np.fliplr(state)) == 0) == 1
                ):
                    return 0.5

                if (
                    np.count_nonzero(state[i, :] == 3 - player) == 2
                    and np.count_nonzero(state[i, :] == 0) == 1
                ):
                    return -0.5
                if (
                    np.count_nonzero(state[:, i] == 3 - player) == 2
                    and np.count_nonzero(state[:, i] == 0) == 1
                ):
                    return -0.5
                if (
                    np.count_nonzero(np.diag(state) == 3 - player) == 2
                    and np.count_nonzero(np.diag(state) == 0) == 1
                ):
                    return -0.5
                if (
                    np.count_nonzero(np.diag(np.fliplr(state)) == 3 - player) == 2
                    and np.count_nonzero(np.diag(np.fliplr(state)) == 0) == 1
                ):
                    return -0.5
            return 0

    def generate_all_states(self):
        states = []
        for values in np.ndindex(3, 3, 3, 3, 3, 3, 3, 3, 3):
            state = np.array(values).reshape(3, 3)
            if 0 <= np.count_nonzero(state == 1) - np.count_nonzero(state == 2) <= 1:
                states.append(state)
        return states

    def current_player(self, state):
        if np.count_nonzero(state == 1) == np.count_nonzero(state == 2):
            return 1
        else:
            return 2

    def train(self, gamma=1, theta=1e-12):
        while True:
            delta = 0
            for state in tqdm(self.states, desc="Value Iteration", leave=False):
                if self.terminal(state):
                    continue

                v = self.V[tuple(map(tuple, state))]
                values = []
                actions = self.possible_actions(state)

                for action in actions:
                    next_state = self.result(state, action, self.current_player(state))
                    reward_val = self.reward(
                        next_state, self.current_player(next_state)
                    )
                    values.append(
                        reward_val + gamma * self.V[tuple(map(tuple, next_state))]
                    )

                if self.current_player(state) == self.PLAYER_X:
                    self.V[tuple(map(tuple, state))] = max(values)
                    self.policy[tuple(map(tuple, state))] = actions[np.argmax(values)]
                else:
                    self.V[tuple(map(tuple, state))] = min(values)
                    self.policy[tuple(map(tuple, state))] = actions[np.argmin(values)]

                delta = max(delta, abs(v - self.V[tuple(map(tuple, state))]))

            if delta < theta:
                break

        for state in self.states:
            if self.terminal(state):
                continue
            values = []
            actions = self.possible_actions(state)

            for action in actions:
                next_state = self.result(state, action, self.current_player(state))
                reward_val = self.reward(next_state, self.current_player(next_state))
                values.append(
                    reward_val + gamma * self.V[tuple(map(tuple, next_state))]
                )

            if self.current_player(state) == self.PLAYER_X:
                self.policy[tuple(map(tuple, state))] = actions[np.argmax(values)]
            else:
                self.policy[tuple(map(tuple, state))] = actions[np.argmin(values)]

    def best_action(self):
        state = self.board
        return self.policy[tuple(map(tuple, state))]

    def save_policy(self):
        with open(self.POLICY_FILE, 'wb') as f:
            pickle.dump(self.policy, f)

    def load_policy(self):
        with open(self.POLICY_FILE, 'rb') as f:
            self.policy = pickle.load(f)
