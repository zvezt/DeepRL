#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from ..utils import *
import time
from .BaseAgent import *

class DQNActor(BaseActor):
    def __init__(self, config, master_network):
        BaseActor.__init__(self, config, master_network)
        self.config = config

    def _transition(self):
        if self._state is None:
            self._state = self._task.reset()
        config = self.config
        q_values = self._network(config.state_normalizer(np.stack([self._state])))
        q_values = to_np(q_values).flatten()
        if self._total_steps < config.exploration_steps \
                or np.random.rand() < config.random_action_prob():
            action = np.random.randint(0, len(q_values))
        else:
            action = np.argmax(q_values)
        next_state, reward, done, info = self._task.step(action)
        entry = [self._state, action, reward, next_state, done, info]
        if done:
            next_state = self._task.reset()
        self._state = next_state
        return entry

class DQNAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config

        self.network = config.network_fn()
        self.network.share_memory()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(self.network.parameters())

        self.replay = config.replay_fn()
        self.actor = DQNActor(config, self.network)
        self.actor.start()

        self.network.to(Config.DEVICE)
        self.target_network.to(Config.DEVICE)

        self.episode_reward = 0
        self.episode_rewards = []

        self.total_steps = 0

    def close(self):
        close_obj(self.replay)
        close_obj(self.actor)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(np.stack([state]))
        q = self.network(state)
        action = np.argmax(to_np(q).flatten())
        self.config.state_normalizer.unset_read_only()
        return action

    def step(self):
        config = self.config
        state, action, reward, next_state, done, _ = self.actor.step()
        self.episode_reward += reward
        self.total_steps += 1
        reward = config.reward_normalizer(reward)
        if done:
            self.episode_rewards.append(self.episode_reward)
            self.episode_reward = 0
        self.replay.feed([state, action ,reward, next_state, int(done)])

        if self.total_steps > self.config.exploration_steps \
                and self.total_steps % self.config.sgd_update_frequency == 0:
            experiences = self.replay.sample()
            states, actions, rewards, next_states, terminals = experiences
            states = self.config.state_normalizer(states)
            next_states = self.config.state_normalizer(next_states)
            q_next = self.target_network(next_states).detach()
            if self.config.double_q:
                best_actions = torch.argmax(self.network(next_states), dim=-1)
                q_next = q_next[range_tensor(q_next.size(0)), best_actions]
            else:
                q_next = torch_max(q_next, 1)
            terminals = tensor(terminals)
            rewards = tensor(rewards)
            q_next = self.config.discount * q_next * (1 - terminals)
            q_next.add_(rewards)
            actions = tensor(actions).long()
            q = self.network(states)
            q = q[range_tensor(q.size(0)), actions]
            loss = (q_next - q).pow(2).mul(0.5).mean()
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
            self.optimizer.step()

        if self.total_steps % self.config.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())