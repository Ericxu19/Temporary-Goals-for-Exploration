import mrl
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import os
from mrl.replays.online_her_buffer import OnlineHERBuffer

class EntropyGainPredictor(mrl.Module):
  """
  Predicts expected entropy gain for a behaviour goal using a learned model
  TODO: Also include version for cross entropy
  """

  def __init__(self, batch_size = 50, history_length = 200, optimize_every=250, log_every=5000):
    super().__init__(
      'entropy_gain_predictor',
      required_agent_modules=[
        'env', 'replay_buffer', 'entropy_gain_model'
      ],
      locals=locals())
    self.log_every = log_every
    self.batch_size = batch_size
    self.history_length = history_length
    self.optimize_every = optimize_every
    self.opt_steps = 0

  def _setup(self):
    super()._setup()
    assert isinstance(self.replay_buffer, OnlineHERBuffer)
    assert self.env.goal_env
    self.n_envs = self.env.num_envs
    self.optimizer = torch.optim.Adam(self.entropy_gain_model.model.parameters())

  def _optimize(self):
    self.opt_steps += 1

    if len(self.replay_buffer.buffer.trajectories) > self.batch_size and self.opt_steps % self.optimize_every == 0:
      trajs = self.replay_buffer.buffer.sample_trajectories(self.batch_size, group_by_buffer=True, from_m_most_recent=self.history_length)
      bg_item_idx = self.replay_buffer.buffer.BUFF.items.index('buffer_bg')
      ag_item_idx = self.replay_buffer.buffer.BUFF.items.index('buffer_ag')

      # TODO: Assuming that fixed behaviour goal for the full episode, but that's not always true
      # TODO: Account for that above
      behav_goals = np.array([t[0] for t in trajs[bg_item_idx]])
      same_bg = [((t[0] - t[-1])**2).sum() == 0.0 for t in trajs[bg_item_idx]]
      if False in same_bg:
        import ipdb; ipdb.set_trace()
      achieved_goals = np.array([t[-5:] for t in trajs[ag_item_idx]])
      batch_size, episode_length, goal_dim = achieved_goals.shape
      achieved_goals = np.reshape(achieved_goals, [self.batch_size * episode_length, goal_dim])

      # Compute Monte Carlo estimates of the entropy gain as targets for the entropy_gain_model
      beta = 1 / len(self.replay_buffer.buffer)
      sampled_ag_entr_new = self.ag_kde.evaluate_elementwise_entropy(achieved_goals, beta=beta)
      sampled_ag_entr_old = self.ag_kde.evaluate_elementwise_entropy(achieved_goals, beta=0.)
      sampled_ag_entr_gain = sampled_ag_entr_new - sampled_ag_entr_old
      sampled_ag_entr_gain = np.reshape(sampled_ag_entr_gain, [self.batch_size, episode_length])
      sampled_ag_entr_gain = sampled_ag_entr_gain.mean(axis=1)
      sampled_ag_entr_gain /= beta # Normalize by beta # TODO: Get rid of this part if not necessary

      targets = self.torch(sampled_ag_entr_gain[:, np.newaxis]) # Shape: batch_size, 1
      inputs = self.torch(behav_goals)

      # In training the discriminator, do we need L2 regularization or something?
      # outputs here have not been passed through sigmoid
      outputs = self.entropy_gain_model(inputs) # Shape: batch_size, 1
      
      # TODO: Grad norm clipping? Prob don't need
      loss = F.mse_loss(outputs, targets)

      if hasattr(self, 'logger'):
        self.logger.add_histogram('entropy_gain_predictions', outputs, self.log_every)
        self.logger.add_histogram('entropy_gain_targets', targets, self.log_every)

      # optimize
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

  def __call__(self, *states_and_maybe_goals):
    """Input / output are numpy arrays"""
    states = np.concatenate(states_and_maybe_goals, -1)
    return self.numpy(self.entropy_gain_model(self.torch(states)))

  def save(self, save_folder : str):
    path = os.path.join(save_folder, self.module_name + '.pt')
    torch.save({
      'opt_state_dict': self.optimizer.state_dict()
    }, path)

  def load(self, save_folder : str):
    path = os.path.join(save_folder, self.module_name + '.pt')
    checkpoint = torch.load(path)
    self.optimizer.load_state_dict(checkpoint['opt_state_dict'])


class MixtureDensityNetworkPredictor(mrl.Module):
  """
  Samples achieved goal prediction for a given behaviour goal.
  Trained the MDN to maximize likelihood of predicting the achieved goals
  """

  def __init__(self, batch_size = 50, history_length = 200, optimize_every=250, log_every=5000, num_ags_in_eps=5):
    super().__init__(
      'mdn_predictor',
      required_agent_modules=[
        'env', 'replay_buffer', 'mixture_density_network', 'ag_kde'
      ],
      locals=locals())
    self.log_every = log_every
    self.batch_size = batch_size
    self.history_length = history_length
    self.optimize_every = optimize_every
    self.opt_steps = 0
    self.num_ags_in_eps = num_ags_in_eps

  def _setup(self):
    super()._setup()
    assert isinstance(self.replay_buffer, OnlineHERBuffer)
    assert self.env.goal_env
    self.n_envs = self.env.num_envs
    self.optimizer = torch.optim.Adam(self.mixture_density_network.model.parameters())

  def _optimize(self):
    self.opt_steps += 1

    if len(self.replay_buffer.buffer.trajectories) > self.batch_size and self.opt_steps % self.optimize_every == 0:
      trajs = self.replay_buffer.buffer.sample_trajectories(self.batch_size, group_by_buffer=True, from_m_most_recent=self.history_length)
      bg_item_idx = self.replay_buffer.buffer.BUFF.items.index('buffer_bg')
      ag_item_idx = self.replay_buffer.buffer.BUFF.items.index('buffer_ag')

      # TODO: Assuming that fixed behaviour goal for the full episode, but that's not always true
      behav_goals = np.array([t[-self.num_ags_in_eps:] for t in trajs[bg_item_idx]]) 
      achieved_goals = np.array([t[-self.num_ags_in_eps:] for t in trajs[ag_item_idx]]) 

      batch_size, episode_length, goal_dim = achieved_goals.shape
      # Compute loss 
      achieved_goals = np.reshape(achieved_goals, [self.batch_size * episode_length, goal_dim])
      behav_goals = np.reshape(behav_goals, [self.batch_size * episode_length, goal_dim])

      # Normalize Batch with KDE
      achieved_goals = (achieved_goals - self.ag_kde.kde_sample_mean) / self.ag_kde.kde_sample_std
      behav_goals = (behav_goals - self.ag_kde.kde_sample_mean) / self.ag_kde.kde_sample_std

      inputs = self.torch(behav_goals)
      outputs = self.torch(achieved_goals)
      pi, normal = self.mixture_density_network(inputs) 
      samples = torch.sum(pi.sample().unsqueeze(2) * normal.sample(), dim=1)
      loglik = normal.log_prob(outputs.unsqueeze(1).expand_as(normal.loc))
      loglik = torch.sum(loglik, dim=2)
      loss = torch.mean(-torch.logsumexp(torch.log(pi.probs) + loglik, dim=1))

      if hasattr(self, 'logger'):
        self.logger.add_scalar('cond_density_nll', loss, self.log_every)
        self.logger.add_histogram('cond_density_pred', outputs, self.log_every)
        self.logger.add_histogram('cond_density_target', achieved_goals, self.log_every)

      # optimize
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

  def __call__(self, states, num_samples=1):
    """Input / output are numpy arrays"""
    # Normalize the states
    states = (states - self.ag_kde.kde_sample_mean) / self.ag_kde.kde_sample_std

    pi, normal = self.mixture_density_network(self.torch(states)) # these are predicted success %
    samples = torch.sum(pi.sample().unsqueeze(2) * normal.sample(), dim=1)

    # Apply de-normalization
    samples = (self.numpy(samples) * self.ag_kde.kde_sample_std) + self.ag_kde.kde_sample_mean
    return samples

  def save(self, save_folder : str):
    path = os.path.join(save_folder, self.module_name + '.pt')
    torch.save({
      'opt_state_dict': self.optimizer.state_dict()
    }, path)

  def load(self, save_folder : str):
    path = os.path.join(save_folder, self.module_name + '.pt')
    checkpoint = torch.load(path)
    self.optimizer.load_state_dict(checkpoint['opt_state_dict'])