import random
import time
from collections import deque

import flappy_bird_gymnasium
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda")  # CUDA 12.1, torch 2.5.1+cu121
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


env = gym.make("FlappyBird-v0")
obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n

low_t = torch.as_tensor(
    env.observation_space.low, device=device, dtype=torch.float32
)
high_t = torch.as_tensor(
    env.observation_space.high, device=device, dtype=torch.float32
)


class DQN(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(obs_shape, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )
        nn.init.uniform_(self.layers[-1].weight, -1e-3, 1e-3)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, x):
        return self.layers(x)


dq_net = DQN(obs_dim, n_actions).to(device)

target_net = DQN(obs_dim, n_actions).to(device)
target_net.load_state_dict(dq_net.state_dict())
target_net.eval()


optimizer = optim.AdamW(dq_net.parameters(), lr=1e-3, fused=True)
criterion = nn.MSELoss()
scaler = torch.amp.GradScaler("cuda")


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.ptr = 0
        self.full = False
        self.s, self.a, self.r, self.ns, self.d = [], [], [], [], []

    def append(self, s, a, r, ns, d):
        if len(self.s) < self.capacity:
            self.s.append(s)
            self.a.append(a)
            self.r.append(r)
            self.ns.append(ns)
            self.d.append(d)
        else:
            self.s[self.ptr] = s
            self.a[self.ptr] = a
            self.r[self.ptr] = r
            self.ns[self.ptr] = ns
            self.d[self.ptr] = d
            self.full = True
        self.ptr = (self.ptr + 1) % self.capacity

    def sample(self, batch_size):
        idxs = torch.randint(0, len(self), (batch_size,), device=device)
        states = torch.stack([self.s[i.item()] for i in idxs], dim=0)
        actions = torch.stack([self.a[i.item()] for i in idxs], dim=0)
        rewards = torch.stack([self.r[i.item()] for i in idxs], dim=0)
        next_states = torch.stack([self.ns[i.item()] for i in idxs], dim=0)
        dones = torch.stack([self.d[i.item()] for i in idxs], dim=0)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self.capacity if self.full else self.ptr


buffer_size = 20000
min_replay_size = 5000
batch_size = 128
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.99
n_epochs = 500
frame_update_skip = 4

update_freq = 500
step = 0

replay_buffer = ReplayBuffer(capacity=buffer_size)
scores = deque(maxlen=10)
threshold = 20


def to_state(x_np):
    return torch.clamp(
        torch.as_tensor(x_np, device=device, dtype=torch.float32),
        min=low_t,
        max=high_t,
    )


for epoch in range(n_epochs):
    state_np, _ = env.reset()
    state = to_state(state_np)
    terminated = truncated = False
    total_reward = 0

    while not (terminated or truncated):
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
            action_t = torch.tensor(action, device=device, dtype=torch.long)
        else:
            with torch.no_grad(), torch.amp.autocast(
                "cuda", dtype=torch.float16
            ):
                action_t = dq_net(state.unsqueeze(0)).argmax(dim=1)
            action = int(action_t.item())

        next_state_np, reward, terminated, truncated, _ = env.step(action)
        next_state = to_state(next_state_np)

        finished = terminated or truncated
        shaped_reward = reward + 0.1
        total_reward += reward 

        reward_t = torch.tensor(shaped_reward, device=device, dtype=torch.float32)
        finished_t = torch.tensor(
            float(finished), device=device, dtype=torch.float32
        )

        replay_buffer.append(
            state, action_t.squeeze(0), reward_t, next_state, finished_t
        )
        state = next_state
        step += 1

        if len(replay_buffer) >= max(batch_size, min_replay_size) and step % frame_update_skip == 0:
            states, actions, rewards, next_states, finisheds = (
                replay_buffer.sample(batch_size)
            )

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                q_values = (
                    dq_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                )
                with torch.no_grad():
                    next_actions = dq_net(next_states).argmax(dim=1)
                    next_q_values = (
                        target_net(next_states)
                        .gather(1, next_actions.unsqueeze(1))
                        .squeeze(1)
                    )
                    target = rewards + gamma * next_q_values * (1.0 - finisheds)
                loss = criterion(q_values, target)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(dq_net.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            if step % update_freq == 0:
                target_net.load_state_dict(dq_net.state_dict())

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    scores.append(total_reward)
    avg_score = float(torch.tensor(scores, dtype=torch.float32).mean().item())
    print(
        f"Epoch {epoch+1}, Avg_Score: {avg_score:.3f}, epsilon: {epsilon:.3f}"
    )
    if avg_score >= threshold and len(scores) == 10:
        print("DQN has reached the goal, stopping training!")
        break

env.close()


# ---- Evaluation ----

test_env = gym.make("FlappyBird-v0", render_mode="human")
dq_net.eval()
state_np, _ = test_env.reset()
state = to_state(state_np)
terminated = truncated = False
total_reward = 0.0

while not (terminated or truncated):
    test_env.render()
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        action_t = dq_net(state.unsqueeze(0)).argmax(dim=1)
    action = int(action_t.item())

    next_state_np, reward, terminated, truncated, _ = test_env.step(action)
    state = to_state(next_state_np)
    total_reward += reward
    time.sleep(0.02)

print(f"Test result: Score = {total_reward:.3f}")
test_env.close()
