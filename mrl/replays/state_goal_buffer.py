"""
TODO: HAVE BARELY STARTED ON THIS. 
"""

import mrl
import gym
from mrl.replays.core.shared_buffer import SharedMemoryTrajectoryBuffer as Buffer
import numpy as np
import torch


class OnlineHERBuffer(mrl.Module):

  def __init__(
      self,
      size: int,  # maximum size (not including hindsight exps)
      save_buffer: bool = False,  # should buffer be saved to disk along with Agent?
  ):
    """
    Buffer that supports sampling both plain experiences, and state-goal relabeled experiences. 
    Does online hindsight relabeling.
    Replaces the old combo of ReplayBuffer + HERBuffer.
    """

    super().__init__(
        'replay_buffer', required_agent_modules=['env'], locals=locals())

    self.size = size
    self.save_buffer = save_buffer
    self.goal_space = None

  def _setup(self):
    env = self.env
    if type(env.observation_space) == gym.spaces.Dict:
      observation_space = env.observation_space.spaces["observation"]
      self.goal_space = env.observation_space.spaces["desired_goal"]
    else:
      observation_space = env.observation_space

    items = [("state", observation_space.shape),
             ("action", env.action_space.shape), ("reward", (1,)),
             ("next_state", observation_space.shape), ("done", (1,))]

    if self.goal_space is not None:
      items += [("previous_ag", self.goal_space.shape), # for reward shaping
                ("ag", self.goal_space.shape), 
                ("dg", self.goal_space.shape)]

    self.buffer = Buffer(self.size, items)
    self._subbuffers = [[] for _ in range(self.env.num_envs)]
    self.n_envs = self.env.num_envs

    hindsight_mode = self.config.get('her')
    if 'future_' in hindsight_mode:
      _, fut = hindsight_mode.split('_')
      self.fut = float(fut) / (1. + float(fut))
      self.act = 0.
      self.ach = 0.
    elif 'futureactual_' in hindsight_mode:
      _, fut, act = hindsight_mode.split('_')
      non_hindsight_frac = 1. / (1. + float(fut) + float(act))
      self.fut = float(fut) * non_hindsight_frac
      self.act = float(act) * non_hindsight_frac
      self.ach = 0.
    elif 'futureachieved_' in hindsight_mode:
      _, fut, ach = hindsight_mode.split('_')
      non_hindsight_frac = 1. / (1. + float(fut) + float(ach))
      self.fut = float(fut) * non_hindsight_frac
      self.act = 0.
      self.ach = float(ach) * non_hindsight_frac
    elif 'rfaa_' in hindsight_mode:
      _, real, fut, act, ach = hindsight_mode.split('_')
      denom = (float(real) + float(fut) + float(act) + float(ach))
      self.fut = float(fut) / denom
      self.act = float(act) / denom
      self.ach = float(ach) / denom
    else:
      self.fut = 0.
      self.act = 0.
      self.ach = 0.

  def _process_experience(self, exp):
    if getattr(self, 'logger'):
      self.logger.add_tabular('Replay buffer size', len(self.buffer))
    done = np.expand_dims(exp.done, 1)  # format for replay buffer
    reward = np.expand_dims(exp.reward, 1)  # format for replay buffer
    action = exp.action

    if self.goal_space:
      state = exp.state['observation']
      next_state = exp.next_state['observation']
      previous_achieved = exp.state['achieved_goal']
      achieved = exp.next_state['achieved_goal']
      if hasattr(self, 'ag_curiosity') and self.ag_curiosity.current_goals is not None:
        goal = self.ag_curiosity.current_goals
      else:
        goal = exp.state['desired_goal']
      for i in range(self.n_envs):
        self._subbuffers[i].append([
            state[i], action[i], reward[i], next_state[i], done[i], previous_achieved[i], achieved[i],
            goal[i]
        ])
    else:
      state = exp.state
      next_state = exp.next_state
      for i in range(self.n_envs):
        self._subbuffers[i].append(
            [state[i], action[i], reward[i], next_state[i], done[i]])

    for i in range(self.n_envs):
      if exp.trajectory_over[i]:
        trajectory = [np.stack(a) for a in zip(*self._subbuffers[i])]
        self.buffer.add_trajectory(*trajectory)
        self._subbuffers[i] = []

  def sample(self, batch_size):
    if self.goal_space:
      if self.config.get('her'):
        fut_batch_size, act_batch_size, ach_batch_size, real_batch_size = np.random.multinomial(
            batch_size, [self.fut, self.act, self.ach, 1.])

        # Sample the real batch
        states, actions, rewards, next_states, dones, previous_ags, ags, goals =\
            self.buffer.sample(real_batch_size)

        # Sample the future batch and relabel it
        states_fut, actions_fut, _, next_states_fut, dones_fut, previous_ags_fut, ags_fut, _, goals_fut =\
          self.buffer.sample_future(fut_batch_size)
        rewards_fut = self.env.compute_reward(ags_fut, goals_fut, None).reshape(-1, 1)

        # Sample the actual batch and relabel it
        states_act, actions_act, _, next_states_act, dones_act, previous_ags_act, ags_act, _, goals_act =\
          self.buffer.sample_actual(act_batch_size)
        rewards_act = self.env.compute_reward(ags_act, goals_act, None).reshape(-1, 1)

        # Sample the achieved batch and relabel it
        states_ach, actions_ach, _, next_states_ach, dones_ach, previous_ags_ach, ags_ach, _, goals_ach =\
          self.buffer.sample_achieved(ach_batch_size)
        rewards_ach = self.env.compute_reward(ags_ach, goals_ach, None).reshape(-1, 1)

        # Concatenate the two
        states = np.concatenate([states, states_fut, states_act, states_ach], 0)
        actions = np.concatenate([actions, actions_fut, actions_act, actions_ach], 0)
        rewards = rewards_O = np.concatenate([rewards, rewards_fut, rewards_act, rewards_ach], 0).astype(np.float32)
        goals = np.concatenate([goals, goals_fut, goals_act, goals_ach], 0)

        if self.config.get('sparse_reward_shaping'):
          previous_ags = np.concatenate([previous_ags, previous_ags_fut, previous_ags_act, previous_ags_ach], 0)
          current_ags  = np.concatenate([ags, ags_fut, ags_act, ags_ach], 0)
          previous_phi = np.linalg.norm(previous_ags - goals, axis=1, keepdims=True)
          current_phi  = np.linalg.norm(current_ags - goals, axis=1, keepdims=True)
          rewards_F = self.config.gamma * current_phi - previous_phi
          rewards_O = rewards
          rewards = rewards_O + self.config.sparse_reward_shaping * rewards_F

        next_states = np.concatenate(
            [next_states, next_states_fut, next_states_act, next_states_ach], 0)
        if self.config.get('never_done'):
          dones = np.zeros_like(rewards, dtype=np.float32)
        elif self.config.get('first_visit_succ'):
          dones = np.round(rewards_O + 1.)
        else:
          dones = np.concatenate([dones, dones_fut, dones_act, dones_ach], 0)

      else:
        states, actions, rewards, next_states, dones, goals =\
                                                    self.buffer.sample(batch_size)
      states = np.concatenate((states, goals), -1)
      next_states = np.concatenate((next_states, goals), -1)

    else:
      states, actions, rewards, next_states, dones = self.buffer.sample(
          batch_size)

    if hasattr(self, 'state_normalizer'):
      states = self.state_normalizer(states, update=False).astype(np.float32)
      next_states = self.state_normalizer(
          next_states, update=False).astype(np.float32)

    return (self.torch(states), self.torch(actions),
            self.torch(rewards), self.torch(next_states),
            self.torch(dones))

  def __len__(self):
    return len(self.buffer)

  def save(self, save_folder):
    if self.save_buffer:
      state = self.buffer._get_state()
      with open(os.path.join(save_folder, "{}_buffer.pickle".format(self.module_name)), 'wb') as f:
        pickle.dump(state, f)

  def load(self, save_folder):
    with open(os.path.join(save_folder, "{}_buffer.pickle".format(self.module_name)), 'rb') as f:
      state = pickle.load(f)
    self.buffer._set_state(state)