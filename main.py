import flappy_bird_gymnasium
import gymnasium as gym
import torch

from src import PPOAgent, frames_to_video, train_model

device = "cuda"
env = gym.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=False)

input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
num_episodes = 10000
max_steps = 1000

lr = 1e-3
gamma = 0.90
epsilon = 0.2

agent = PPOAgent(input_dim, output_dim, lr, gamma, epsilon, device=device)

train_model(env, agent, num_episodes, max_steps, device)

state = env.reset()[0]
frames = []

while True:
    with torch.no_grad():
        action = agent.select_action(state)
        state_next, r, done, truncated, info = env.step(action)
        frames.append(env.render())
        state = state_next
        if done or truncated:
            break

frames_to_video(frames, fps=24)
