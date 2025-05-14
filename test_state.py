from console import FlappyBirdEnv
from my_agent_0th import *

env = FlappyBirdEnv(config_file_path='config.yml', show_screen=False, level=1)
env.reset()
state = env.get_state()

# Test BUILD_STATE
features = BUILD_STATE(state)
print("Feature vector:", features)

# Simulate a step
action = 1  # do_nothing
env.step(action)
next_state = env.get_state()

# Test REWARD
r = REWARD(state, next_state)
print("Reward:", r)

agent = MyAgent()

state = env.get_state()
a = agent.choose_action(state, env.get_action_table())
print("Action chosen:", a)
print("Current memory:", agent.storage[-1])