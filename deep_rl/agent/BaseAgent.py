#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import torch
import numpy as np
import torch.multiprocessing as mp
from collections import deque
import sys
from ..utils import *
import copy

class BaseAgent:
    def __init__(self, config):
        self.config = config
        self.eval_env = self.config.eval_env
        if self.eval_env is not None:
            self.evaluation_state = self.eval_env.reset()
            self.evaluation_return = 0

    def save(self, filename):
        torch.save(self.network.state_dict(), filename)

    def load(self, filename):
        state_dict = torch.load(filename, map_location=lambda storage, loc: storage)
        self.network.load_state_dict(state_dict)

    def eval_step(self, state):
        pass

    def deterministic_episode(self):
        env = self.config.eval_env
        state = env.reset()
        total_rewards = 0
        while True:
            action = self.eval_step(state)
            state, reward, done, _ = env.step(action)
            total_rewards += reward
            if done:
                break
        return total_rewards

    def evaluation_episodes(self):
        rewards = []
        for ep in range(self.config.evaluation_episodes):
            rewards.append(self.deterministic_episode())
        self.config.logger.info('evaluation episode return: %f(%f)' % (
            np.mean(rewards), np.std(rewards) / np.sqrt(len(rewards))))

class BaseActor(mp.Process):
    STEP = 0
    RESET = 1
    EXIT = 2
    SPECS = 3
    NETWORK = 4
    CACHE = 5

    def __init__(self, config):
        mp.Process.__init__(self)
        self.config = config
        self.__pipe, self.__worker_pipe = mp.Pipe()

        self._state = None
        self._task = None
        self._network = None
        self._total_steps = 0

    def run(self):
        torch.cuda.is_available()
        config = self.config
        random_seed()
        seed = np.random.randint(0, sys.maxsize)
        self._task = config.task_fn()
        self._task.seed(seed)
        cache = deque([], maxlen=config.cache_len)
        while True:
            if self.__worker_pipe.poll():
                op, data = self.__worker_pipe.recv()
            else:
                op = self.STEP
            if op == self.STEP:
                # if len(cache) == 0:
                #     cache.append(self._transition())
                # self.__worker_pipe.send(cache.popleft())
                if self._network is None:
                    continue
                self.__worker_pipe.send(self._transition())
            # elif op == self.CACHE:
            #     while len(cache) < config.cache_len:
            #         cache.append(self._transition())
            elif op == self.EXIT:
                close_obj(self._task)
                self.__worker_pipe.close()
                return
            elif op == self.NETWORK:
                self._network = data
            else:
                raise Exception('Unknown command')

    def _transition(self):
        pass

    def step(self):
        # self.__pipe.send([self.STEP, None])
        return self.__pipe.recv()

    def close(self):
        self.__pipe.send([self.EXIT, None])
        self.__pipe.close()

    def set_network(self, net):
        self.__pipe.send([self.NETWORK, net])

    # def cache(self):
    #     self.__pipe.send([self.CACHE, None])
