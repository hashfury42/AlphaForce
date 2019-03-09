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


class PGModel(nn.Module):
    def __init__(self):
        super(PGModel, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, 2)
        self.saved_log_probs = []
        self.rewards = []

    def predict(self, x):
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

    def generate_action(self, observation):
        state = torch.from_numpy(observation).float().unsqueeze(0)
        prob_list = self._model.predict(state)
        m = Categorical(prob_list)
        action = m.sample()
        self._model.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def train_model(self):
        R = 0
        policy_loss = []
        returns = []
        optimizer = optim.Adam(pg_model.parameters(), lr=1e-2)
        eps = np.finfo(np.float32).eps.item()
        for r in self._model.rewards[::-1]:
            R = r + self._gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)
        for log_prob, R in zip(self._model.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()
        del self._model.rewards[:]
        del self._model.saved_log_probs[:]


if __name__ == "__main__":

    # The model for Vanilla PGAgent
    pg_model = PGModel()

    # The environment for the agent
    env = gym.make('CartPole-v1')
    env.seed(42)
    torch.manual_seed(42)

    # pg agent
    agent = PGAgent(gamma=0.99, model=pg_model, processor=None)
    agent.fit(env, 1000, 10000, 10)