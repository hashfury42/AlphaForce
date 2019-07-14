#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from core.agent import Agent
from core.memory import PGMemory
from core.processor import RewardProcessor


# reference
# https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py


class PGModel(nn.Module):
    def __init__(self, input_dim=0, output_dim=0):
        super(PGModel, self).__init__()
        self.affine1 = nn.Linear(input_dim, 64)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(64, output_dim)
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


class PGAgent(Agent):
    """The basic Vanilla REINFORCE agent.

    """

    def __init__(self, gamma=0.99, **kwargs):
        super(PGAgent, self).__init__(**kwargs)
        self._gamma = gamma
        self.optimizer = optim.Adam(pg_model.parameters(), lr=1e-2)

    def generate_action(self, observation):
        state = torch.from_numpy(observation).float().unsqueeze(0)
        prob_list = self._model.forward(state)
        m = Categorical(prob_list)
        action = m.sample()
        self._memory.log_probs.append(m.log_prob(action))
        return action.item()

    def _train_model(self):
        # Prepare train data.
        policy_loss = []
        self._memory.generate_batch(self._memory.capacity)
        returns = \
            self._processor.get_discounted_rewards(self._memory.batch.reward, gamma=0.99)
        for log_prob, reward in zip(self._memory.log_probs, returns):
            policy_loss.append(-log_prob * reward)

        # Train ...
        self.optimizer.zero_grad()

        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

        # Clean the memory.
        self._memory.reset()

    def _on_episode_begin(self, env):
        state_current, ep_reward, state_anchor = env.reset(), 0, None
        return state_current, ep_reward, state_anchor


if __name__ == "__main__":
    # The environment for the agent
    env = gym.make('CartPole-v1')
    env.seed(42)
    torch.manual_seed(42)

    # The model for Vanilla PGAgent
    pg_model = PGModel(input_dim=env.observation_space.shape[0],
                       output_dim=env.action_space.n)
    # The memory
    memory = PGMemory(10000)

    # The processor
    processor = RewardProcessor()

    # PG agent
    agent = PGAgent(gamma=0.99,
                    model=pg_model,
                    processor=processor,
                    memory=memory)

    agent.fit(env, n_episode=1000, n_max_actions=10000, is_render=False, log_interval=10)
