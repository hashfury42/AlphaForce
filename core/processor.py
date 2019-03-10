#!/usr/bin/python
# -*- coding: utf-8 -*-
import abc
import numpy as np
import torch


class Processor(object):
    """This is the abstract base model class for agents.
    """
    __metaclass__ = abc.ABCMeta



class RewardProcessor(Processor):

    def __init__(self):
        self._eps = np.finfo(np.float32).eps.item()

    def get_discounted_rewards(self, rewards, gamma=0.99):
        """Takes gamma of rewards and computes discounted reward
        e.g. f([1, 1, 1], 0.99) -> [2.9701, 1.99, 1]
        """
        discounted_rewards = []
        R = 0
        for r in rewards[::-1]:
            R = r + gamma * R
            discounted_rewards.insert(0, R)
        returns = torch.tensor(discounted_rewards)
        returns = (returns - returns.mean()) / (returns.std() + self._eps)
        return returns