#!/usr/bin/python
# -*- coding: utf-8 -*-
import abc


class Agent(object):
    """This is the abstract base class for all agents.

    To implement your own agent, you have to implement the following methods:

    - `initialize`
    - `fit`
    - `load_model`
    - `save_model`

    Args
        model
        processor
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, model=None, processor=None):
        self._model = model
        self._processor = processor

    def fit(self, env, n_episode, n_max_actions, is_render=False, log_interval=100):
        running_reward = 0
        for episode in range(n_episode):
            state, ep_reward = env.reset(), 0
            for t in range(1, n_max_actions):  # Don't infinite loop while learning
                action = self.generate_action(state)
                state, reward, done, _ = env.step(action)
                if is_render: env.render()
                self._model.rewards.append(reward)
                ep_reward += reward
                if done:
                    break
            running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
            self.train_model()
            if episode % log_interval == 0:
                print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                    episode, ep_reward, running_reward))
            if running_reward > env.spec.reward_threshold:
                print("Solved! Running reward is now {} and "
                      "the last episode runs to {} time steps!".format(running_reward, t))
                break

    @abc.abstractmethod
    def generate_action(self, observation):
        """Watch an observation from the environment and returns the action

        :param observation:
        :return:
        """
        pass

    @abc.abstractmethod
    def train_model(self):
        """

        :return:
        """
        pass

    def filter_reward(self, reward):
        """

        :param reward:
        :return:
        """
        pass

    def load_model(self, file_path):
        """Loads the model of an agent from an HDF5 file.

        # Args
            file_path (str): The path to the HDF5 file.
        """
        pass

    def save_model(self, file_path, overwrite=False):
        """Saves the model of an agent as an HDF5 file.

        # Arguments
            file_path (str): The path to where the weights should be saved.
            overwrite (boolean): If `False` and `filepath` already exists, raises an error.
        """
        pass