#!/usr/bin/python
# -*- coding: utf-8 -*-
import gym
import math
import random
import traceback
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from core.agent import Agent
from core.memory import DQNMemory
from core.config import device
from core.processor import CartPoleObservationProcessor
from core.model import Model


# reference
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html


class DQN(nn.Module):

    def __init__(self, h, w):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, 2)  # 448 or 512

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


class DQNModel(Model):

    def __init__(self, h, w):
        self.policy_net = DQN(h, w).to(device)
        self.target_net = DQN(h, w).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters())


class DQNAgent(Agent):

    def __init__(self, **kwargs):
        super(DQNAgent, self).__init__(**kwargs)
        self.BATCH_SIZE = 128
        self.GAMMA = 0.999
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 200
        self.TARGET_UPDATE = 10
        self.steps_done = 0

    def generate_action(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self._model.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(2)]], device=device, dtype=torch.long)

    def _train_model(self):
        if len(self._memory) < self.BATCH_SIZE:
            return
        self._memory.generate_batch(self.BATCH_SIZE, is_sample=True)

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                self._memory.batch.next_state)),
                                      device=device, dtype=torch.uint8)
        non_final_next_states = \
            torch.cat([s for s in self._memory.batch.next_state if s is not None])
        state_batch = torch.cat(self._memory.batch.state)
        action_batch = torch.cat(self._memory.batch.action)
        reward_batch = torch.cat(self._memory.batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self._model.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self._model.target_net(non_final_next_states).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self._model.optimizer.zero_grad()
        loss.backward()
        for param in self._model.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self._model.optimizer.step()

    def _on_episode_begin(self, env):
        env.reset()
        ep_reward = 0
        last_screen = self._processor.get_screen(env)
        current_screen = self._processor.get_screen(env)
        state_current = current_screen - last_screen
        state_anchor = current_screen
        return state_current, ep_reward, state_anchor

    def _on_episode_end(self, episode):
        # Update the target network, copying all weights and biases in DQN
        if episode % self.TARGET_UPDATE == 0:
            self._model.target_net.load_state_dict(self._model.policy_net.state_dict())

    def _on_action_end(self, env, state_next, reward, done, state_anchor):
        reward = torch.tensor([reward], device=device)
        # Observe new state
        last_screen = state_anchor
        current_screen = self._processor.get_screen(env)
        if not done:
            state_next = current_screen - last_screen
        else:
            state_next = None
        state_anchor = current_screen
        return state_next, reward, done, state_anchor


if __name__ == "__main__":
    # The processor
    processor = CartPoleObservationProcessor()

    # The environment for the agent
    try:
        from pyvirtualdisplay import Display
        display = Display(visible=0, size=(1400, 900))
        display.start()
    except Exception as e:
        pass
        # traceback.print_exc()

    plt.ion()
    env = gym.make('CartPole-v0').unwrapped
    env.seed(42)
    torch.manual_seed(42)
    env.reset()
    plt.figure()
    plt.imshow(processor.get_screen(env).cpu().squeeze(0).permute(1, 2, 0).numpy(),
               interpolation='none')
    plt.title('Example extracted screen')
    plt.show()
    _, _, screen_height, screen_width = processor.get_screen(env).shape

    # The model for DQN Agent
    dqn_model = DQNModel(screen_height, screen_width)

    # The memory
    memory = DQNMemory(10000)

    # dqn agent
    agent = DQNAgent(model=dqn_model, processor=processor, memory=memory)

    agent.fit(env, n_episode=1000, n_max_actions=10000, is_render=False, log_interval=10)
