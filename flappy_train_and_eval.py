# flappy_train_and_eval.py

import argparse
import numpy as np
import matplotlib.pyplot as plt
from console import FlappyBirdEnv
from my_agent_1 import MyAgent


def train_agent(level: int = 1, episodes: int = 1000, show_screen: bool = False):
    env = FlappyBirdEnv(
        config_file_path='config.yml',
        show_screen=False,
        level=level,
        game_length=10
    )
    agent = MyAgent(show_screen=show_screen, mode='train')
    print(f"Agent is using device: {agent.device}")

    scores   = []
    mileages = []
    best_score = -float('inf')

    for episode in range(episodes):
        env.play(player=agent)
        print(f"Episode {episode}: Score = {env.score}, Mileage = {env.mileage}")

        scores.append(env.score)
        mileages.append(env.mileage)

        # Save best‐performing model
        if env.score > best_score:
            best_score = env.score
            agent.save_model('my_model.ckpt')

    return scores, mileages


def evaluate_agent(level: int = 1, episodes: int = 10, show_screen: bool = False):
    env = FlappyBirdEnv(
        config_file_path='config.yml',
        show_screen=False,
        level=level,
        game_length=10
    )
    agent = MyAgent(show_screen=False, load_model_path='my_model.ckpt', mode='eval')

    scores = []
    for episode in range(episodes):
        env.play(player=agent)
        scores.append(env.score)

    print("\n[Evaluation Result]")
    print(f"Max Score: {np.max(scores)}")
    print(f"Mean Score: {np.mean(scores):.2f}")

    return scores


def plot_progress(scores, mileages, save_path="training_progress.png"):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(scores)
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title("Score vs Episode")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(mileages, color="orange")
    plt.xlabel("Episode")
    plt.ylabel("Mileage")
    plt.title("Mileage vs Episode")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--level', type=int, default=1)
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--show_screen', action='store_true', help="Show the game screen during training/evaluation")
    args = parser.parse_args()

    if not args.eval_only:
        scores, mileages = train_agent(level=args.level, episodes=args.episodes, show_screen=args.show_screen)
        plot_progress(scores, mileages)

    evaluate_agent(level=args.level, episodes=10, show_screen=args.show_screen)