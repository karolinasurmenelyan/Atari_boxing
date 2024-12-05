import os
import gymnasium as gym
import numpy as np
import random
import imageio
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import ale_py


def train_and_save_model(train_model, MODEL_PATH, vec_env):

    print("Vectorized environment created:", vec_env)

    if not train_model:
        if os.path.exists(MODEL_PATH):
            model = PPO.load(MODEL_PATH)
            print("Model loaded from", MODEL_PATH)
        else:
            raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found.")
    else:
        model = PPO("CnnPolicy", vec_env, verbose=1)

    if train_model:
        print("Training the model...")
        model.learn(total_timesteps=1000000)
        model.save(MODEL_PATH)
        print("Model saved to", MODEL_PATH)
    return model


def record_video(env, out_directory, model, fps=30, max_steps=1000):
    images = []
    obs = env.reset()
    rewards_sum = 0

    for _ in range(max_steps):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        rewards_sum += rewards[0]

        img = env.render("rgb_array")
        images.append(img)

        if dones[0]:
            break

    imageio.mimsave(out_directory, [np.array(image) for image in images], fps=fps)
    print(f"Video saved to {out_directory}. Total reward: {rewards_sum}")


def main2():
    gym.register_envs(ale_py)

    # it False if you want to upload model from your pc
    train_model = False
    MODEL_PATH = "models/ppo_cartpole.zip"
    vec_env = make_vec_env("ALE/Boxing-v5", n_envs=4)
    model = train_and_save_model(train_model, MODEL_PATH, vec_env)

    video_path = "Agent_PPO.mp4"
    record_video(vec_env, video_path, model=model)
    print(f"Video saved to {video_path}. Open it with your favorite video player.")

main2()