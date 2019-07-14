#!/usr/bin/python
# -*- coding: utf-8 -*-
import abc


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
        """
        running_reward = 0
        for episode in range(n_episode):
            # An episode start and environment state initialized.
            state_current, ep_reward, state_anchor = self._on_episode_begin(env)

            # Actions start.
            for t in range(1, n_max_actions):
                ################################################################
                # Generate an action.
                ################################################################
                self._on_action_start()
                action = self.generate_action(state_current)
                state_next, r, done, _ = env.step(self._processor.transform_action(action))
                state_next, r, done, state_anchor = self._on_action_end(env, state_next, r, done, state_anchor)
                ################################################################
                # Push transition state into memory.
                ################################################################
                self._memory.push(state_current, action, state_next, r)
                state_current = state_next
                ep_reward += self._processor.transform_reward(r)

                if is_render: env.render()
                if done: break

            # The model will be trained only after the whole epoch ends.
            self._train_model()

            # Show logs
            running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
            self._log(episode, running_reward, ep_reward, log_interval)
            if running_reward > env.spec.reward_threshold:
                print("Solved! Running reward is now {} !".format(running_reward))
                break

            # an episode end
            self._on_episode_end(episode)

    @abc.abstractmethod
    def generate_action(self, observation):
        """Watch an observation from the environment and returns the action.
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
    def _train_model(self):
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

    def _log(self, episode, running_reward, ep_reward, log_interval):
        if episode % log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                episode, ep_reward, running_reward))
