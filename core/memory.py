#!/usr/bin/python
# -*- coding: utf-8 -*-
import abc
import random
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class Memory(object):
    """Basic abstract class for memory.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.batch = None

    @abc.abstractmethod
    def push(self, *args):
        pass

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def generate_batch(self, batch_size, is_sample=False):
        """Transform the memory to Transition Aggregation Format.
        e.g.
        [Transition(state=3, action=6, next_state=7, reward=1),
        Transition(state=5, action=6, next_state=7, reward=1)]
        to Transition(state=(3, 5), action=(6, 6), next_state=(7, 7), reward=(1, 1))
        """
        if is_sample:
            self.batch = Transition(*zip(*self.sample(batch_size)))
        else:
            self.batch = Transition(*zip(*self.memory[0:batch_size]))

    def reset(self):
        self.memory = []
        self.batch = None

    def __len__(self):
        return len(self.memory)


class PolicyMemory(Memory):
    """Memory for Policy Gradients methods.
    """

    def __init__(self, capacity, **kwargs):
        super(PolicyMemory, self).__init__(capacity)
        self.log_probs = []

    def push(self, *args):
        self.memory.append(Transition(*args))

    def reset(self):
        self.memory = []
        self.log_probs = []


class ReplayMemory(Memory):
    """Memory for Model based methods.
    """

    def __init__(self, capacity, **kwargs):
        super(ReplayMemory, self).__init__(capacity)
        self.position = 0

    def push(self, *args):
        """Saves a transition.
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
