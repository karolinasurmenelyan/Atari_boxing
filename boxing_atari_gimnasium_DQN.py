# -*- coding: utf-8 -*-
"""boxing_Atari_Gimnasium.ipynb

Original file is located at
    https://colab.research.google.com/drive/1CB0xcgZ6u_B5s9N-H7py8YIsgn0KHcHg

https://gymnasium.farama.org/environments/atari/boxing/
"""


from pyvirtualdisplay import Display
import numpy as np

from collections import deque

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import gymnasium as gym
import random
import os
import tqdm
import ale_py
import pickle as pickle
from tqdm.notebook import tqdm

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

# import gym
import gymnasium as gym
from tqdm import tqdm

import imageio
# from IPython.display import HTML
from base64 import b64encode
import ale_py


class DQN_agent(nn.Module):
    def __init__(self, s_size, a_size, model_dir="models"):
        super(DQN_agent, self).__init__()
        # Training parameters
        self.n_training_episodes = 100  # Total training episodes
        self.n_evaluation_episodes = 50
        self.learning_rate = 0.001          # Learning rate
        self.state_size = s_size
        self.action_size = a_size
        self.model_dir = model_dir

        # Environment parameters
        self.max_steps = 10                # Max steps per episode
        self.gamma = 0.95                  # Discounting rate

        # Exploration parameters
        self.max_epsilon = 1.0               # Exploration probability at start
        self.min_epsilon = 0.01               # Minimum exploration probability
        self.decay_rate = 0.0001             # Exponential decay rate for exploration prob
        self.memory = deque(maxlen=2000)
        self.batch_size = 32

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.model = self._build_model()
        self.target_model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(2, 2), activation='relu', input_shape=(80, 80, 4)))  # Вход: (80, 80, 4)
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))  # Полносвязный слой
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss=MeanSquaredError())
        return model

    def forward(self, x):
        return self.model(x)

    def save_model(self, episode):
        model_path = os.path.join(self.model_dir, f"dqn_model_ep{episode}.keras")
        self.model.save(model_path)
        print(f"Model saved at {model_path}")

    def load_model(self, model_name="dqn_model_ep65.keras"):
        model_path = os.path.join(self.model_dir, model_name)
        if os.path.exists(model_path):
            self.model.load_weights(model_path)
            self.target_model.set_weights(self.model.get_weights())
            print(f"Model loaded from {model_path}")
        else:
            print(f"Model {model_path} not found. Training from scratch.")

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, terminated, truncated, info):
        self.memory.append((state, action, reward, next_state, terminated, truncated, info))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, terminated, truncated, info in minibatch:
            target = reward
            if not terminated:
                next_state = np.concatenate(next_state, axis=-1).reshape(1, 80, 80, 4)  # Вход как 4 последовательных кадра
                target += self.gamma * np.amax(self.target_model.predict(next_state)[0])
            state = np.concatenate(state, axis=-1).reshape(1, 80, 80, 4)  # Вход как 4 последовательных кадра
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.max_epsilon > self.min_epsilon:
            self.max_epsilon -= self.decay_rate

    def greedy_policy(self, state):
        q_values = self.model.predict(state)
        action = np.argmax(q_values)
        return action

    def epsilon_greedy_policy(self, state, epsilon):
        random_num = np.random.uniform(0, 1)
        if random_num > epsilon:
            action = self.greedy_policy(state.reshape(1, 80, 80, 4))
        else:
            action = np.random.randint(0, self.action_size)
        return action


def preprocess_state(state):
    state = state[35:195]
    state = state[::2, ::2, :]
    state = np.mean(state, axis=2)
    state = state.astype(np.float32)
    state = state / 255.0
    state = state.reshape(80, 80, 1)
    return state


def train(env, DQN_agent):
    for episode in tqdm(range(DQN_agent.n_training_episodes)):
        if episode % 3 == 0:
            DQN_agent.update_target_network()
        epsilon = DQN_agent.min_epsilon + (DQN_agent.max_epsilon - DQN_agent.min_epsilon) * np.exp(-DQN_agent.decay_rate * episode)
        state, info = env.reset()
        state = preprocess_state(state)
        state_buffer = deque(maxlen=4)
        for _ in range(4):
            state_buffer.append(state)

        terminated = False
        truncated = False

        for step in range(DQN_agent.max_steps):
            state_input = np.concatenate(list(state_buffer), axis=-1).reshape(1, 80, 80, 4)  # Собираем 4 последовательных кадра
            action = DQN_agent.epsilon_greedy_policy(state_input, epsilon)
            new_state, reward, terminated, truncated, info = env.step(action)
            new_state = preprocess_state(new_state)
            state_buffer.append(new_state)
            new_state_input = np.concatenate(list(state_buffer), axis=-1).reshape(1, 80, 80, 4)  # Собираем 4 кадра для следующего состояния
            DQN_agent.remember(state_input, action, reward, new_state_input, terminated, truncated, info)

            if terminated or truncated:
                break

            DQN_agent.replay()

        # Сохраняем модель каждые 10 эпизодов
        if episode % 10 == 0:
            DQN_agent.save_model(episode)

    # Финальное сохранение модели
    DQN_agent.save_model(DQN_agent.n_training_episodes)



def evaluate_agent(env, max_steps, n_eval_episodes, DQN_agent, epsilon=0):
    """
    Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
    :param env: The evaluation environment
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param epsilon: Exploration parameter, set to 0 for greedy policy during evaluation.
    """
    episode_rewards = []

    for episode in range(n_eval_episodes):
        print(episode)
        state, _ = env.reset()
        state = preprocess_state(state)

        # Create a state buffer to store the last 4 frames
        state_buffer = deque(maxlen=4)
        for _ in range(4):
            state_buffer.append(state)

        terminated = False
        truncated = False
        total_rewards_ep = 0

        for step in range(max_steps):
            # Concatenate 4 frames into the input state
            state_input = np.concatenate(list(state_buffer), axis=-1).reshape(1, 80, 80, 4)
            action = DQN_agent.epsilon_greedy_policy(state_input, epsilon=epsilon)
            print("State shape", state_input.shape)

            new_state, reward, terminated, truncated, _ = env.step(action)
            new_state = preprocess_state(new_state)
            state_buffer.append(new_state)  # Update buffer with new state
            total_rewards_ep += reward

            if terminated or truncated:
                break

        episode_rewards.append(total_rewards_ep)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward


def record_video(env, DQN_agent, out_directory, fps=30):
    images = []

    state, _ = env.reset()
    state = preprocess_state(state)

    # Создаем буфер для хранения 4 последних состояний
    state_buffer = deque(maxlen=4)
    for _ in range(4):
        state_buffer.append(state)

    img = env.render()
    images.append(img)
    rewards_sum = 0

    terminated = False
    truncated = False

    while not (terminated or truncated):
        # Подготовка входного состояния из 4 кадров
        state_input = np.concatenate(list(state_buffer), axis=-1).reshape(1, 80, 80, 4)
        action = DQN_agent.epsilon_greedy_policy(state_input, epsilon=0)  # Получаем действие от агента
        new_state, reward, terminated, truncated, _ = env.step(action)
        rewards_sum += reward

        img = env.render()
        images.append(img)

        # Обновляем буфер состояний
        new_state = preprocess_state(new_state)
        state_buffer.append(new_state)

    imageio.mimsave(out_directory, [np.array(image) for image in images], fps=fps)
    print(f"Total reward: {rewards_sum}")



#
# def show_video(video_path, video_width = 500):
#
#   video_file = open(video_path, "r+b").read()
#
#   video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"
#   return HTML(f"""<video width={video_width} controls><source src="{video_url}"></video>""")
#
# show_video("Agent.mp4")


def print_info(env):

    print("_____OBSERVATION SPACE_____ \n")
    print("Observation Space", env.observation_space)
    print("Sample observation", env.observation_space.sample())  # Get a random observation

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("\n _____ACTION SPACE_____ \n")
    print("Action Space Shape", env.action_space.n)
    print("Action Space Sample", env.action_space.sample())  # Take a random action

    state_space = env.observation_space.shape[0]
    print("There are ", state_space, " possible states")

    action_space = env.action_space.n
    print("There are ", action_space, " possible actions")
    return state_space, action_space


def main():
    # virtual_display = Display(visible=0, size=(1400, 900))
    # virtual_display.start()

    gym.register_envs(ale_py)

    env = gym.make("ALE/Boxing-v5", render_mode="rgb_array")
    state_space, action_space = print_info(env=env)

    DQN_agent_ = DQN_agent(state_space, action_space)

    model_name = "dqn_model_ep65.keras"
    if os.path.exists(os.path.join(DQN_agent_.model_dir, model_name)):
        DQN_agent_.load_model(model_name)
    else:
        train(env, DQN_agent_)

    mean_reward, std_reward = evaluate_agent(env=env,
                                             max_steps=DQN_agent_.max_steps,
                                             n_eval_episodes=DQN_agent_.n_evaluation_episodes,
                                             DQN_agent=DQN_agent_)
    print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")
    record_video(env, DQN_agent_, "Agent_DQN.mp4")

if __name__ == "__main__":
    main()