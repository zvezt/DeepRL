import torch
import numpy as np
from ..utils import *
from ..component import *
import sys
import torch.multiprocessing as mp

class Command:
    WEIGHTS = 0
    TRANSITIONS = 1
    SAMPLE = 2
    EXIT = 3
    DONE = 4
    EPISODE_REWARDS = 5

class BaseActor(mp.Process):
    def __init__(self, config, network, learner_pipe, replay_pipe):
        mp.Process.__init__(self)
        self.config = config
        self.learner_pipe = learner_pipe
        self.replay_pipe = replay_pipe
        self.network = network
        self.episode_reward = 0

    def run(self):
        self.network.to(Config.DEVICE)
        config = self.config
        random_seed()
        seed = np.random.randint(0, sys.maxsize)
        self.task = config.task_fn()
        self.task.seed(seed)
        self.state = self.task.reset()
        self.total_steps = 0
        while True:
            entries = []
            for i in range(config.rollout_length):
                entries.append(self.transition())
            self.replay_pipe.send([Command.TRANSITIONS, entries])
            op, data = self.replay_pipe.recv()
            # if op == Command.DONE:
            #     self.learner_pipe.send([Command.WEIGHTS, None])
            #     op, state_dict = self.learner_pipe.recv()
            #     self.network.load_state_dict(state_dict)
            # else:
            #     raise Exception('BaseActor: unknown command')
            if self.total_steps > 1000:
                print('STOP')
                while True: pass

    def transition(self):
        raise Exception('_transition not implemented')

class BaseReplay(mp.Process):
    def __init__(self, config, learner_pipe, actor_pipes):
        mp.Process.__init__(self)
        self.memory_size = config.memory_size
        self.batch_size = config.batch_size
        self.cache_len = 1
        self.learner_pipe = learner_pipe
        self.actor_pipes = actor_pipes
        self.config = config
        self.total_stpes = 0
        self.episode_returns = []

    def run(self):
        torch.cuda.is_available()
        replay = Replay(self.memory_size, self.batch_size)
        cache = deque([], maxlen=self.cache_len)

        def sample():
            batch_data = replay.sample()
            batch_data = [tensor(x) for x in batch_data]
            cache.append(batch_data)

        cached = None
        cur_actor_pipe = 0
        while True:
            if self.learner_pipe.poll():
                op, data = self.learner_pipe.recv()
                if op == Command.EXIT:
                    return
                elif op == Command.EPISODE_REWARDS:
                    self.learner_pipe.send([Command.EPISODE_REWARDS, self.episode_returns])
                    self.episode_returns = []
                elif op == Command.SAMPLE:
                    if replay.size() < self.config.min_memory_size:
                        self.learner_pipe.send([Command.DONE, None])
                        continue
                    if len(cache) == 0:
                        sample()
                        cached = cache.popleft()
                    self.learner_pipe.send([Command.DONE, cached])
                    # self.learner_pipe.send([Command.DONE, cache.popleft()])
                    # while len(cache) < self.cache_len:
                    #     sample()
                else:
                    raise Exception('BaseReplay: unknown command')
            actor_pipe = self.actor_pipes[cur_actor_pipe]
            cur_actor_pipe = (cur_actor_pipe + 1) % self.config.num_workers
            if replay.size() > self.config.min_memory_size:
                continue
            if actor_pipe.poll():
                op, data = actor_pipe.recv()
                if op == Command.TRANSITIONS:
                    for transition, episode_return in data:
                        replay.feed(transition)
                        if episode_return is not None:
                            self.episode_returns.append(episode_return)
                    actor_pipe.send([Command.DONE, None])
                    self.config.total_steps.value += len(data)
                else:
                    raise Exception('BaseReplay: unknown command')

class BaseLearner:
    def __init__(self, config):
        leaner_actor_pipes = [mp.Pipe() for _ in range(config.num_workers)]
        replay_actor_pipes = [mp.Pipe() for _ in range(config.num_workers)]
        learner_replay_pipe = mp.Pipe()
        self.network = config.network_fn()
        self.actor_networks = [config.network_fn() for _ in range(config.num_workers)]
        for actor_network in self.actor_networks:
            actor_network.load_state_dict(self.network.state_dict())

        self.actors = [config.actor_fn(config, self.actor_networks[i],
                                       leaner_actor_pipes[i][1], replay_actor_pipes[i][1])
                       for i in range(config.num_workers)]
        self.replay = config.replay_fn(config, learner_replay_pipe[1],
                                       [pipe[0] for pipe in replay_actor_pipes])
        self.actors_pipes = [pipe[0] for pipe in leaner_actor_pipes]
        self.replay_pipe = learner_replay_pipe[0]
        self.config = config
        self.config.total_steps = mp.Value('i')
        self.cur_actor = 0
        self.sgd_steps = 0

        for actor in self.actors: actor.start()
        self.replay.start()

    @property
    def total_steps(self):
        return self.config.total_steps.value

    @property
    def episode_rewards(self):
        self.replay_pipe.send([Command.EPISODE_REWARDS, None])
        op, data = self.replay_pipe.recv()
        return data

    @episode_rewards.setter
    def episode_rewards(self, _):
        pass

    def sgd_update(self, experiences):
        raise Exception('not implemented')

    def step(self):
        t0 = time.time()
        self.replay_pipe.send([Command.SAMPLE, None])
        op, experiences = self.replay_pipe.recv()
        t1 = time.time()
        if experiences is not None:
            self.sgd_update(experiences)
            self.sgd_steps += 1
        t2 = time.time()

        actor_pipe = self.actors_pipes[self.cur_actor]
        self.cur_actor = (self.cur_actor + 1) % self.config.num_workers
        if actor_pipe.poll():
            op, data = actor_pipe.recv()
            if op == Command.WEIGHTS:
                actor_pipe.send([Command.WEIGHTS, self.network.state_dict()])
        ts = [t0, t1, t2, time.time()]
        print(np.diff(ts) / (ts[-1] - ts[0]))

    def run(self):
        config = self.config
        t0 = time.time()
        while True:
            self.step()
            if self.sgd_steps and config.log_interval and not self.sgd_steps % config.log_interval:
                returns = self.episode_rewards
                if not len(returns):
                    returns.append(0)
                config.logger.info('total steps %d, returns %.2f/%.2f/%.2f/%.2f (mean/median/min/max), %.2f steps/s' % (
                    self.total_steps, np.mean(returns), np.median(returns),
                    np.min(returns), np.max(returns),
                    config.log_interval / (time.time() - t0)))
                t0 = time.time()
