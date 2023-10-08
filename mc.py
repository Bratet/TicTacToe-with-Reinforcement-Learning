
from utils import TicTacToe
import random
from collections import defaultdict
from tqdm import tqdm
import numpy as np

class MontoCarlo(TicTacToe):
    def __init__(self):
        super().__init__()
        self.train()

    def behavior_policy(self, state, action=None):
  
        epsilon = 0.1
        n = len(self.actions())

        if action:
            best_act = self.best_action_for_state(state)
            if action == best_act:
                return 1 - epsilon + epsilon/n
            else:
                return epsilon/n
        else:
            if np.random.rand() < epsilon:
                return random.choice(self.actions())
            else:
                return self.best_action_for_state(state)

    def best_action_for_state(self, state):
        actions = self.actions()
        best_act = max(actions, key=lambda action: self.Q[(state, action)], default=None)
        return best_act if best_act else random.choice(actions)

    def generate_episode(self, policy):
        episode = []
        game = TicTacToe()

        while not game.terminal():
            s = game.board.tobytes()
            action = policy(s)
            episode.append((s, action))
            game = game.result(action)

        return episode

    def train(self, num_simulations=100000): 
        self.Q = defaultdict(lambda: 1.0)
        self.C = defaultdict(float)

        alpha = 0.9

        for _ in tqdm(range(num_simulations)):
            episode = self.generate_episode(self.behavior_policy)
            G = 0
            W = 1

            if len(episode) > 0:
                final_state, _ = episode[-1]
                G = self.utility(self.PLAYER_X if self.x_is_human else self.PLAYER_O)

            for t in reversed(range(len(episode))):
                state, action = episode[t]

                self.C[(state, action)] += W
                self.Q[(state, action)] += alpha * (G - self.Q[(state, action)]) 
                best_action = self.best_action_for_state(state)

                if action != best_action:
                    break
                W *= 1.0 / self.behavior_policy(state, action)

    def best_action(self):
        state = self.board.tobytes()
        return self.best_action_for_state(state)
