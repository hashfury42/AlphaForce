#!/usr/bin/python
# -*- coding: utf-8 -*-
import unittest
from core.memory import PolicyMemory, Transition


class PolicyMemoryTestCase(unittest.TestCase):
    def test_push(self):
        mem = PolicyMemory(100)
        mem.push(1, 2, 3, 1)
        mem.push(5, 6, 7, 1)
        mem.push(3, 6, 7, 1)
        print(mem.memory)
        print(mem.memory)
        transitions = mem.sample(2)
        print(transitions)
        batch = Transition(*zip(*transitions))
        print(batch)
        print(batch.reward)
        mem.generate_batch(100)
        mem.get_discounted_rewards(gamma=0.99)
        print(mem.discounted_rewards)


if __name__ == '__main__':
    unittest.main()
