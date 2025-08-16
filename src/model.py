import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 32)
        self.actor = nn.Linear(32, output_dim)
        self.critic = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.actor(x), self.critic(x)


# Proximal Policy Optimization Agent
class PPOAgent:
    def __init__(
        self,
        input_dim,
        output_dim,
        lr=0.001,
        gamma=0.99,
        epsilon=0.2,
        device="cpu",
    ):
        self.actor_critic = ActorCritic(input_dim, output_dim).to(device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.device = device

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        logits, _ = self.actor_critic(state)
        action_probs = F.softmax(logits, dim=-1)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        return action.item()

    def compute_returns(self, rewards):
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        return returns

    def update(self, states, actions, returns, advantages):
        states = torch.tensor(np.array(states), dtype=torch.float32).to(
            self.device
        )
        actions = torch.tensor(actions, dtype=torch.int32).to(self.device)
        returns = returns.clone().detach().to(self.device)
        advantages = advantages.clone().detach().to(self.device)

        logits, values = self.actor_critic(states)
        action_probs = F.softmax(logits, dim=-1)
        action_dist = torch.distributions.Categorical(action_probs)
        log_probs = action_dist.log_prob(actions)

        ratio = torch.exp(log_probs - log_probs.detach())
        surr1 = ratio * advantages
        surr2 = (
            torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        )
        actor_loss = -torch.min(surr1, surr2).mean()

        critic_loss = F.smooth_l1_loss(values.squeeze(), returns)

        total_loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
