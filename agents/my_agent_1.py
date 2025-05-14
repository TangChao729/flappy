import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import random
from collections import deque
from pytorch_mlp import MLPRegression, DEVICE
import argparse

from console import FlappyBirdEnv
STUDENT_ID = 'a1234567'
DEGREE = 'UG'  # or 'PG'


class MyAgent:
    def __init__(self, show_screen=False, load_model_path=None, mode=None):
        # do not modify these
        self.show_screen = show_screen
        self.mode = mode or 'train'  # 'train' or 'eval'
        self.device = DEVICE

        # --- DQN hyperparameters ---
        self.input_dim = 4               # φ has 4 features
        self.output_dim = 2              # [jump, do_nothing]
        self.batch_size = 128             # minibatch size
        self.capacity = 10000            # replay buffer capacity
        self.epsilon = 1.0               # start with full exploration
        self.epsilon_min = 0.05           # final ε
        self.epsilon_decay = 0.995       # per-step decay
        self.gamma = 0.99                # discount factor

        # --- replay buffer ---
        # each entry: dict with keys 'phi', 'action', 'reward', 'q_next'
        self.storage = deque(maxlen=self.capacity)

        # --- Q-networks ---
        self.network  = MLPRegression(input_dim=self.input_dim,
                                      output_dim=self.output_dim,
                                      learning_rate=5e-4).to(self.device)
        self.network2 = MLPRegression(input_dim=self.input_dim,
                                      output_dim=self.output_dim,
                                      learning_rate=5e-4).to(self.device)
        # initialize target net
        MyAgent.update_network_model(self.network2, self.network)

        # internal counter to update target periodically
        self.step_count = 0
        self.target_update_freq = 1000

        # training step counter
        self.training_step    = 0  # added for training frequency control
        self.update_counter   = 0
        self.update_frequency = 1000
        self.discount_factor  = 0.99

        # storage of previous state for reward computation
        self.prev_state = None

        # do not modify this
        if load_model_path:
            self.load_model(load_model_path)

    def BUILD_STATE(self, state: dict) -> np.ndarray:
        """
        φ = [ bird_y_norm, bird_v_norm,
              horiz_dist_to_next_pipe, vert_dist_to_gap_center ]
        """
        bird_x          = state['bird_x']
        bird_y          = state['bird_y']
        bird_velocity   = state['bird_velocity']
        screen_width    = state['screen_width']
        screen_height   = state['screen_height']
        pipes           = state['pipes']

        # find next pipe
        next_pipe       = next((pipe for pipe in pipes if pipe['x'] + pipe['width'] >= bird_x), None)

        if next_pipe is None:
            next_pipe_dist_to_bird = screen_width
            next_pipe_center_y = screen_height / 2
        else:
            next_pipe_dist_to_bird = next_pipe['x'] + next_pipe['width'] - state['bird_x']
            next_pipe_center_y = (next_pipe['top'] + next_pipe['bottom']) / 2

        bird_to_gap_y = bird_y - next_pipe_center_y

        return np.array([
            bird_y / screen_height,
            bird_velocity / 10,
            next_pipe_dist_to_bird / screen_width,
            bird_to_gap_y / screen_height
        ], dtype=np.float32)

    def REWARD(self, prev_state: dict, curr_state: dict) -> float:
        if curr_state['done']:
            if curr_state['done_type'] == 'hit_pipe':
                return -1.0
            elif curr_state['done_type'] == 'offscreen':
                return -2.0
            elif curr_state['done_type'] == 'well_done':
                return 2.0

        reward = 0.02

        # Reward for score increase
        if curr_state.get('score', 0) > prev_state.get('score', 0):
            return reward + 1.5

        # Encourage bird to stay low in the gap
        next_pipe = next((pipe for pipe in curr_state['pipes'] if pipe['x'] + pipe['width'] >= curr_state['bird_x']), None)
        if next_pipe:
            preferred_y = next_pipe['bottom'] - (next_pipe['bottom'] - next_pipe['top']) * 0.33
            dist = abs(curr_state['bird_y'] + 15 - preferred_y) / curr_state['screen_height']
            return reward + 0.1 - 0.3 * dist

        return reward + 0.1

    def choose_action(self, state, action_table):
        phi_t = self.BUILD_STATE(state)
        self.prev_state = state
        self.prev_phi   = phi_t

        actions = [v for k, v in action_table.items() if k != 'quit_game']
        if self.mode == 'train' and np.random.rand() < self.epsilon:
            a_t = np.random.choice(actions)
        else:
            q_values = self.network.predict(phi_t.reshape(1, -1))[0]
            a_t = actions[np.argmax(q_values)]

        if self.mode == 'train':
            self.prev_action = a_t
            self.storage.append({'phi': phi_t, 'action': a_t, 'reward': None, 'q_next': None})
        return a_t

    def receive_after_action_observation(self, state: dict, action_table: dict) -> None:
        if self.mode != 'train':
            return

        phi_t1 = self.BUILD_STATE(state)
        r_t = self.REWARD(self.prev_state, state)
        
        best_a = np.argmax(self.network.predict(phi_t1.reshape(1, -1))[0])
        q_t1 = self.network2.predict(phi_t1.reshape(1, -1))[0][best_a]

        # update the last transition
        self.storage[-1]['reward'] = r_t
        self.storage[-1]['q_next'] = q_t1

        if len(self.storage) < 1000:
            self._sync_networks()
            return

        # sample a minibatch if we have enough
        if len(self.storage) >= self.batch_size:
            batch = random.sample(self.storage, self.batch_size)
            X, Y, W = [], [], []
            for t in batch:
                phi, a, r, q_next = t['phi'], t['action'], t['reward'], t['q_next']
                target_q = self.network.predict(phi.reshape(1, -1))[0]
                target_q[a] = r + self.discount_factor * q_next
                X.append(phi)
                Y.append(target_q)
                W.append([1 if i == a else 0 for i in range(len(target_q))])

            # one gradient step
            self.network.fit_step(np.array(X), np.array(Y), np.array(W))

            # # decay ε
            # self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

            # # periodically update target network
            # self.step_count += 1
            # if self.step_count % self.target_update_freq == 0:
            #     MyAgent.update_network_model(self.network2, self.network)

            self._sync_networks()

        # clear memory at end of episode
        if state['done']:
            self.storage.clear()

    def _sync_networks(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.update_counter += 1
        if self.update_counter % self.update_frequency == 0:
            MyAgent.update_network_model(self.network2, self.network)

    def save_model(self, path: str = 'my_model.ckpt'):
        self.network.save_model(path=path)

    def load_model(self, path: str = 'my_model.ckpt'):
        self.network.load_model(path=path)

    @staticmethod
    def update_network_model(net_to_update: MLPRegression, net_as_source: MLPRegression):
        net_to_update.load_state_dict(net_as_source.state_dict())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--level', type=int, default=1)
    args = parser.parse_args()

    # Training loop (you can tweak episodes, game_length, etc.)
    env = FlappyBirdEnv(config_file_path='config.yml',
                        show_screen=True,
                        level=args.level,
                        game_length=10)
    agent = MyAgent(show_screen=False)
    episodes = 10000
    for episode in range(episodes):
        env.play(player=agent)
        print(f"Episode {episode} — score={env.score}, mileage={env.mileage}")
        agent.save_model(path='my_model.ckpt')

    # Evaluation
    env2 = FlappyBirdEnv(config_file_path='config.yml',
                         show_screen=False,
                         level=args.level)
    agent2 = MyAgent(show_screen=False,
                     load_model_path='my_model.ckpt',
                     mode='eval')
    scores = []
    for _ in range(10):
        env2.play(player=agent2)
        scores.append(env2.score)
    print("Best:", np.max(scores), "Mean:", np.mean(scores))