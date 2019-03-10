#!/usr/bin/python
# -*- coding: utf-8 -*-
import abc
import copy


class Agent(object):
    """This is the abstract base class for all agents.

    Args
        model
        processor
        memory
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, model=None, processor=None, memory=None):
        self._model = model
        self._processor = processor
        self._memory = memory

    def fit(self, env, n_episode=0, n_max_actions=0, is_render=False, log_interval=100):
        """
        :param env:
        :param n_episode:
        :param n_max_actions:
        :param is_render:
        :param log_interval:
        :return:
        """
        running_reward = 0
        for episode in range(n_episode):
            # an episode start and environment state initialized
            state_current, ep_reward, state_anchor = self._on_episode_begin(env)

            # actions start
            for t in range(1, n_max_actions):

                # do an action
                self._on_action_start()
                action = self.generate_action(state_current)
                state_next, reward, done, _ = \
                    env.step(self._processor.transform_action(action))
                state_next, reward, done, state_anchor = \
                    self._on_action_end(env, state_next, reward, done, state_anchor)

                # add state to memory
                self._memory.push(state_current, action, state_next, reward)
                state_current = state_next
                ep_reward += self._processor.transform_reward(reward)

                if is_render:
                    env.render()
                if done:
                    break

            self.train_model()

            # show logs
            running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
            if episode % log_interval == 0:
                print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                    episode, ep_reward, running_reward))
            if running_reward > env.spec.reward_threshold:
                print("Solved! Running reward is now {} and "
                      "the last episode runs to {} time steps!".format(running_reward, t))
                break

            # an episode end
            self._on_episode_end(episode)

    @abc.abstractmethod
    def generate_action(self, observation):
        """Watch an observation from the environment and returns the action.

        :param observation:
        :return:
        """
        pass

    @abc.abstractmethod
    def _on_episode_begin(self, *args):
        """"""
        pass

    def _on_episode_end(self, *args):
        """"""
        pass

    def _on_action_start(self, *args):
        pass

    def _on_action_end(self, env, state_next, reward, done, state_anchor):
        return state_next, reward, done, state_anchor

    @abc.abstractmethod
    def train_model(self):
        """Train the neural network module to optimize the loss function.
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

        # Args
            file_path (str): The path to where the weights should be saved.
            overwrite (boolean): If `False` and `filepath` already exists, raises an error.
        """
        pass