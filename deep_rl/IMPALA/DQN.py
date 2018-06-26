from .prototype import *

class DQNActor(BaseActor):
    def __init__(self, config, network, learner_pipe, replay_pipe):
        BaseActor.__init__(self, config, network, learner_pipe, replay_pipe)

    def transition(self):
        if self.state is None:
            self.state = self.task.reset()
        config = self.config
        q_values = self.network(config.state_normalizer(np.stack([self.state])))
        q_values = to_np(q_values).flatten()
        if self.total_steps < config.exploration_steps \
                or np.random.rand() < config.random_action_prob():
            action = np.random.randint(0, len(q_values))
        else:
            action = np.argmax(q_values)
        next_state, reward, done, info = self.task.step(action)
        self.episode_reward += reward
        reward = config.reward_normalizer(reward)
        entry = [self.state, action, reward, next_state, int(done)]
        self.total_steps += 1
        episode_return = None
        if done:
            next_state = self.task.reset()
            episode_return = self.episode_reward
            self.episode_reward = 0
        self.state = next_state
        return entry, episode_return

class DQNLearner(BaseLearner):
    def __init__(self, config):
        BaseLearner.__init__(self, config)
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.batch_indices = range_tensor(config.batch_size)
        self.network.to(Config.DEVICE)
        self.target_network.to(Config.DEVICE)

    def sgd_update(self, experiences):
        states, actions, rewards, next_states, terminals = experiences
        states = self.config.state_normalizer(states)
        next_states = self.config.state_normalizer(next_states)
        q_next = self.target_network(next_states).detach()
        if self.config.double_q:
            best_actions = torch.argmax(self.network(next_states), dim=-1)
            q_next = q_next[self.batch_indices, best_actions]
        else:
            q_next = torch_max(q_next, 1)
        terminals = tensor(terminals)
        rewards = tensor(rewards)
        q_next = self.config.discount * q_next * (1 - terminals)
        q_next.add_(rewards)
        actions = tensor(actions).long()
        q = self.network(states)
        q = q[self.batch_indices, actions]
        loss = (q_next - q).pow(2).mul(0.5).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
        self.optimizer.step()

        if self.sgd_steps % self.config.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())
