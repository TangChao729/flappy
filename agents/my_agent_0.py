# my_agent.py

import numpy as np
from pytorch_mlp import MLPRegression, DEVICE
from collections import deque
import random

# -------- State Representation --------
def BUILD_STATE(state: dict) -> np.ndarray:
    bird_y = state['bird_y']
    bird_velocity = state['bird_velocity']
    screen_width = state['screen_width']
    screen_height = state['screen_height']
    pipes = state['pipes']

    next_pipe = next((pipe for pipe in pipes if pipe['x'] + pipe['width'] >= state['bird_x']), None)

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

# -------- Reward Function --------
def REWARD(prev_state: dict, curr_state: dict) -> float:
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

# -------- DQN Agent --------
class MyAgent:
    def __init__(self, show_screen=False, load_model_path=None, mode='train'):
        self.show_screen = show_screen
        self.mode = mode
        self.device = DEVICE

        self.network  = MLPRegression(input_dim=4, output_dim=2, learning_rate=4e-4).to(self.device)
        self.network2 = MLPRegression(input_dim=4, output_dim=2, learning_rate=4e-4).to(self.device)
        MyAgent.update_network_model(self.network2, self.network)

        self.storage = deque(maxlen=10000)

        self.epsilon       = 1.0
        self.epsilon_min   = 0.05
        self.epsilon_decay = 0.995

        self.discount_factor  = 0.99
        self.batch_size       = 128
        self.update_frequency = 200
        self.update_counter   = 0
        self.training_step    = 0  # added for training frequency control

        if load_model_path:
            self.load_model(load_model_path)

    def choose_action(self, state, action_table):
        phi_t = BUILD_STATE(state)
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

    def receive_after_action_observation(self, state, action_table):
        if self.mode != 'train':
            return

        phi_t1 = BUILD_STATE(state)
        r_t = REWARD(self.prev_state, state)

        best_a = np.argmax(self.network.predict(phi_t1.reshape(1, -1))[0])
        q_t1 = self.network2.predict(phi_t1.reshape(1, -1))[0][best_a]

        self.storage[-1]['reward'] = r_t
        self.storage[-1]['q_next'] = q_t1

        if len(self.storage) < 1000:
            self._sync_networks()
            return

        self.training_step += 1
        if self.training_step % 4 == 0 and len(self.storage) >= self.batch_size:
            batch = random.sample(self.storage, self.batch_size)
            X, Y, W = [], [], []
            for t in batch:
                phi, a, r, q_next = t['phi'], t['action'], t['reward'], t['q_next']
                target_q = self.network.predict(phi.reshape(1, -1))[0]
                target_q[a] = r + self.discount_factor * q_next
                X.append(phi)
                Y.append(target_q)
                W.append([1 if i == a else 0 for i in range(len(target_q))])

            self.network.fit_step(np.array(X), np.array(Y), np.array(W))

        self._sync_networks()

    def _sync_networks(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.update_counter += 1
        if self.update_counter % self.update_frequency == 0:
            MyAgent.update_network_model(self.network2, self.network)

    def save_model(self, path='my_model.ckpt'):
        self.network.save_model(path)

    def load_model(self, path='my_model.ckpt'):
        self.network.load_model(path)

    @staticmethod
    def update_network_model(net_to_update, net_as_source):
        net_to_update.load_state_dict(net_as_source.state_dict())