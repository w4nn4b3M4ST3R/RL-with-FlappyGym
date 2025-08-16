import numpy as np
import torch


def train_model(env, agent, num_episodes, max_steps, device):
    for episode in range(num_episodes):

        state = env.reset()[0]
        done = False
        episode_rewards = []
        episode_states = []
        episode_actions = []

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)

            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)

            state = next_state

            if done or truncated:
                break

        returns = agent.compute_returns(episode_rewards)

        # Compute advantages
        values = (
            agent.actor_critic(
                torch.tensor(np.array(episode_states), dtype=torch.float32).to(
                    device
                )
            )[1]
            .squeeze()
            .detach()
        )
        advantages = returns - values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-8
        )

        # Update policy
        agent.update(episode_states, episode_actions, returns, advantages)

        total_reward = sum(episode_rewards)

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    env.close()
