# my_agent.py
import numpy as np
from pytorch_mlp import MLPRegression, DEVICE
from collections import deque
import random


INPUT_DIM = 4
OUTPUT_DIM = 2
LEARNING_RATE = 4e-4
STORAGE_SIZE = 10000
EPSILON = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995
DISCOUNT_FACTOR = 0.99
BATCH_SIZE = 128
UPDATE_FREQUENCY = 100

# -------- DQN Agent --------
class MyAgent:
    def __init__(self, show_screen=False, load_model_path=None, mode='train'):
        self.show_screen = show_screen
        self.mode = mode
        self.device = DEVICE

        self.network  = MLPRegression(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, learning_rate=LEARNING_RATE).to(self.device)
        self.network2 = MLPRegression(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, learning_rate=LEARNING_RATE).to(self.device)
        MyAgent.update_network_model(self.network2, self.network)

        self.storage = deque(maxlen=10000)

        self.epsilon       = EPSILON
        self.epsilon_min   = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY

        self.discount_factor  = DISCOUNT_FACTOR
        self.batch_size       = BATCH_SIZE
        self.update_frequency = UPDATE_FREQUENCY
        self.update_counter   = 0
        self.training_step    = 0  # added for training frequency control

        if load_model_path:
            self.load_model(load_model_path)

    # -------- State Representation --------
    def BUILD_STATE(self, state: dict) -> np.ndarray:
        bird_y = state['bird_y']
        bird_width = state['bird_width']
        bird_velocity = state['bird_velocity']
        screen_width = state['screen_width']
        screen_height = state['screen_height']

        next_pipe = self._get_next_pipe(state)

        if next_pipe is None:
            next_pipe_dist_to_bird = screen_width
            next_pipe_center_y = screen_height / 2
        else:
            next_pipe_dist_to_bird = next_pipe['x'] + next_pipe['width'] - state['bird_x'] + bird_width
            next_pipe_center_y = (next_pipe['top'] + next_pipe['bottom']) / 2

        bird_to_gap_y = bird_y - next_pipe_center_y

        return np.array([
            bird_y / screen_height,
            bird_velocity / 10,
            next_pipe_dist_to_bird / screen_width,
            bird_to_gap_y / screen_height
        ], dtype=np.float32).reshape(1, -1)

    # -------- Reward Function --------
    def REWARD(self, prev_state: dict, curr_state: dict) -> float:
        if curr_state['done']:
            if curr_state['done_type'] == 'hit_pipe':
                return -1.0
            elif curr_state['done_type'] == 'offscreen':
                return -2.0
            elif curr_state['done_type'] == 'well_done':
                return 5.0

        reward = 0.05

        # Reward for score increase
        if curr_state.get('score', 0) > prev_state.get('score', 0):
            return reward + 1.5

        # Encourage bird to stay low in the gap
        next_pipe = self._get_next_pipe(curr_state)
        if next_pipe:
            preferred_y = next_pipe['bottom'] - (next_pipe['bottom'] - next_pipe['top']) * 0.33
            dist = abs(curr_state['bird_y'] + 15 - preferred_y) / curr_state['screen_height']
            return reward + 0.1 - 0.3 * dist

        return reward + 0.1
    
    def _get_next_pipe(self, state):
        bird_x = state['bird_x']
        pipes = state['pipes']
        pipe_x = pipes[0]['x'] if pipes else 0
        pipe_width = pipes[0]['width'] if pipes else 0

        next_pipe = next((pipe for pipe in pipes if pipe_x + pipe_width >= bird_x), None)
        return next_pipe

    def choose_action(self, state, action_table):
        phi_t = self.BUILD_STATE(state)
        self.prev_state = state
        self.prev_phi   = phi_t

        actions = [v for k, v in action_table.items() if k != 'quit_game']
        if self.mode == 'train' and np.random.rand() < self.epsilon:
            a_t = np.random.choice(actions)
        else:
            a_t = actions[np.argmax(self.network.predict(phi_t)[0])]

        if self.mode == 'train':
            self.prev_action = a_t
            self.storage.append({'phi': phi_t, 'action': a_t, 'reward': None, 'q_next': None})
        return a_t

    def receive_after_action_observation(self, state, action_table):
        if self.mode != 'train':
            return

        phi_t1 = self.BUILD_STATE(state)
        r_t = self.REWARD(self.prev_state, state)

        best_a = np.argmax(self.network.predict(phi_t1)[0])
        q_t1 = self.network2.predict(phi_t1)[0][best_a]

        self.storage[-1]['reward'] = r_t
        self.storage[-1]['q_next'] = q_t1

        if len(self.storage) < 1000:
            self._sync_networks()
            return

        self.training_step += 1
        if self.training_step % 4 == 0 and len(self.storage) >= self.batch_size:
            batch = random.sample(self.storage, self.batch_size)
            X = np.vstack([t['phi'] for t in batch])       # (batch, 4)
            Y = np.zeros((self.batch_size, OUTPUT_DIM), dtype=np.float32)
            W = np.zeros_like(Y)

            for i, t in enumerate(batch):
                a = t['action']
                target = t['reward'] + self.discount_factor * t['q_next']
                Y[i, a] = target
                W[i, a] = 1.0

            self.network.fit_step(X, Y, W)

        self._sync_networks()

        # clear memory at end of episode
        if state['done']:
            self.storage.clear()

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