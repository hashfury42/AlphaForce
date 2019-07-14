#!/usr/bin/python
# -*- coding: utf-8 -*-
from core.config import device
import abc
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image


class Processor(object):
    """This is the abstract base model class for agents.
    """
    __metaclass__ = abc.ABCMeta

    def transform_reward(self, reward):
        return reward

    def transform_state(self, state):
        return state

    def transform_action(self, action):
        return action


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


class ObservationProcessor(Processor):
    pass


class CartPoleObservationProcessor(ObservationProcessor):

    def __init__(self):
        self.resize = T.Compose([T.ToPILImage(),
                                T.Resize(40, interpolation=Image.CUBIC),
                                T.ToTensor()])

    def get_screen(self, env):
        def get_cart_location(env, screen_width):
            world_width = env.x_threshold * 2
            scale = screen_width / world_width
            return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART
        # Returned screen requested by gym is 400x600x3, but is sometimes larger
        # such as 800x1200x3. Transpose it into torch order (CHW).
        screen = env.render(mode='rgb_array').transpose((2, 0, 1))
        # Cart is in the lower half, so strip off the top and bottom of the screen
        _, screen_height, screen_width = screen.shape
        screen = screen[:, int(screen_height * 0.4):int(screen_height * 0.8)]
        view_width = int(screen_width * 0.6)
        cart_location = get_cart_location(env,screen_width)
        if cart_location < view_width // 2:
            slice_range = slice(view_width)
        elif cart_location > (screen_width - view_width // 2):
            slice_range = slice(-view_width, None)
        else:
            slice_range = slice(cart_location - view_width // 2,
                                cart_location + view_width // 2)
        # Strip off the edges, so that we have a square image centered on a cart
        screen = screen[:, :, slice_range]
        # Convert to float, rescale, convert to torch tensor
        # (this doesn't require a copy)
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        # Resize, and add a batch dimension (BCHW)
        return self.resize(screen).unsqueeze(0).to(device)

    def transform_reward(self, reward):
        return float(reward)

    def transform_action(self, action):
        return int(action)
