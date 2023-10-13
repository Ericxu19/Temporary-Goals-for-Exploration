"""
Curiosity modules for unsupervised exploration
"""

import mrl
import numpy as np
from mrl.replays.online_her_buffer import OnlineHERBuffer
from mrl.utils.misc import softmax, AttrDict
from sklearn.neighbors import KernelDensity
from collections import deque
import copy
import ot

def generate_overshooting_goals(num_proposals, step_amount, direct_overshoots, base_goal):
  base_proposals = np.array([base_goal, base_goal + step_amount])
  if direct_overshoots:
    return base_proposals
  #additional_proposals = base_goal[None] + step_amount[None]*5 + np.random.uniform(
  #    -1.5, 1.5, (num_proposals - 2, step_amount.shape[0])) * (step_amount[None])
  additional_proposals = base_goal[None] + np.random.uniform(
      -1.5, 1.5, (num_proposals - 2, step_amount.shape[0])) * step_amount[None]
  return np.concatenate((base_proposals, additional_proposals), 0)


class AchievedGoalCuriosity(mrl.Module):
  """
    For goal agents only. This module assumes the replay buffer maintains an achieved goal buffer;
    To decide on goals to pursue during exploration, the module samples goals from the achieved goal
    buffer, and chooses the highest scoring (see below) viable (per q-function) goal.  
  """
  def __init__(self, num_sampled_ags=500, max_steps=50, keep_dg_percent=-1e-1, randomize=False, use_qcutoff=True):
    super().__init__('ag_curiosity',
                     required_agent_modules=['env', 'replay_buffer', 'actor', 'critic'],
                     locals=locals())
    self.num_sampled_ags = num_sampled_ags
    self.max_steps = max_steps  #TODO: have this be learned from past trajectories?
    self.keep_dg_percent = keep_dg_percent
    self.randomize = randomize
    self.use_qcutoff = use_qcutoff

  def _setup(self):
    assert isinstance(self.replay_buffer, OnlineHERBuffer)
    assert self.env.goal_env

    self.n_envs = self.env.num_envs
    self.current_goals = None
    self.replaced_goal = np.zeros((self.env.num_envs, ))

    # setup cutoff
    if self.config.gamma < 1.:
      r = min(self.config.gamma, 0.99)
      self.min_min_cutoff = -(1 - r**(self.max_steps * 0.8)) / (1 - r)
    else:
      self.min_min_cutoff = -self.max_steps * 0.8
    self.min_cutoff = max(self.config.initial_cutoff, self.min_min_cutoff)
    self.cutoff = self.min_cutoff

    # go explore + success accounting
    self.go_explore = np.zeros((self.n_envs, 1), dtype=np.float32)
    self.is_success = np.zeros((self.n_envs, 1), dtype=np.float32)
    self.successes_deque = deque(maxlen=10)  # for dynamic cutoff
    self.successes = []
    self.swap = 4
    self.swapcheck = [1. ,2. ,3.]

    #keep track of step for momentum
    self.step = np.zeros((self.n_envs, 1))
    self.past_goals = []

  def _manage_resets_and_success_behaviors_new(self, experience, close):
    """ Manage (1) end of trajectory, (2) early resets, (3) go explore and overshot goals """
    reset_idxs, overshooting_idxs, overshooting_proposals = [], [], []
    if self.step is None:
      self.step = np.zeros((self.n_envs, experience.next_state['achieved_goal'][1].shape[0]))
    for i, over in enumerate(experience.trajectory_over):
      if over:  # if over update it
        self.current_goals[i] = experience.reset_state['desired_goal'][i]
        self.replaced_goal[i] = 0.
        if np.random.random() < (self.go_explore[i] * self.config.go_reset_percent):
          reset_idxs.append(i)
      
      if not over:
        if close[i]: #if not over and success, modify go_explore; maybe overshoot goal?
          if self.go_explore[i] ==0:
            self.is_success[i] += 1.
            #print(experience.next_state['achieved_goal'][i] - experience.state['achieved_goal'][i])
            self.step[i] = experience.next_state['achieved_goal'][i] - experience.state['achieved_goal'][i]
          self.go_explore[i] += 1.
          if self.go_explore[i] > 0 and np.count_nonzero(experience.next_state['achieved_goal'][i] - experience.state['achieved_goal'][i])!= 0:
            self.step[i] = self.step[i]*0.9+ (experience.next_state['achieved_goal'][i] - experience.state['achieved_goal'][i])*0.1
        #print(self.go_explore)
        if not self.config.get('never_done') and np.random.random() < self.config.overshoot_goal_percent and self.go_explore[i] != 0:
          step_amount = self.step[i]

          overshooting_idxs.append(i)
          overshooting_proposals.append(
              generate_overshooting_goals(self.num_sampled_ags, step_amount, self.config.direct_overshoots,
                                          experience.next_state['achieved_goal'][i]))
        
    return reset_idxs, overshooting_idxs, np.array(overshooting_proposals)

  def _manage_resets_and_success_behaviors(self, experience, close):
    """ Manage (1) end of trajectory, (2) early resets, (3) go explore and overshot goals """
    reset_idxs, overshooting_idxs, overshooting_proposals = [], [], []

    for i, over in enumerate(experience.trajectory_over):
      if over:  # if over update it
        self.current_goals[i] = experience.reset_state['desired_goal'][i]
        self.replaced_goal[i] = 0.
        if np.random.random() < (self.go_explore[i] * self.config.go_reset_percent):
          reset_idxs.append(i)

      if not over and close[i]:  # if not over and success, modify go_explore; maybe overshoot goal?
        self.is_success[i] += 1.
        self.go_explore[i] += 1.

        if not self.config.get('never_done') and np.random.random() < self.config.overshoot_goal_percent:
          step_amount = experience.next_state['achieved_goal'][i] - experience.state['achieved_goal'][i]
          overshooting_idxs.append(i)
          overshooting_proposals.append(
              generate_overshooting_goals(self.num_sampled_ags, step_amount, self.config.direct_overshoots,
                                          self.current_goals[i]))

    return reset_idxs, overshooting_idxs, np.array(overshooting_proposals)

  def _overshoot_goals(self, experience, overshooting_idxs, overshooting_proposals):
    #score the proposals
    num_proposals = overshooting_proposals.shape[1]
    num_idxs = len(overshooting_idxs)
    states = np.tile(experience.reset_state['observation'][overshooting_idxs, None, :], (1, num_proposals, 1))
    states = np.concatenate((states, overshooting_proposals), -1).reshape(num_proposals * num_idxs, -1)

    bad_q_idxs, q_values = [], None
    if self.use_qcutoff:
      if hasattr(self, 'state_normalizer'):
          states = self.state_normalizer(states)
      q_values = self.compute_q(states)
      q_values = q_values.reshape(num_idxs, num_proposals)
      bad_q_idxs = q_values < self.cutoff
    goal_values = self.score_goals(overshooting_proposals, AttrDict(q_values=q_values, states=states))
    #print(str(np.count_nonzero(bad_q_idxs)))
    
    
    if self.config.dg_score_multiplier > 1. and self.dg_kde.ready:
      dg_scores = self.dg_kde.evaluate_log_density(overshooting_proposals.reshape(num_proposals * num_idxs, -1))
      dg_scores = dg_scores.reshape(num_idxs, num_proposals)
      goal_values[dg_scores > -np.inf] *= self.config.dg_score_multiplier
    if self.use_qcutoff:
      goal_values[bad_q_idxs] = q_values[bad_q_idxs] * -1e-8

    chosen_idx = np.argmin(goal_values, axis=1)
    chosen_idx = np.eye(num_proposals)[chosen_idx]  # shape(sampled_ags) = n_envs x num_proposals
    chosen_ags = np.sum(overshooting_proposals * chosen_idx[:, :, None], axis=1)  # n_envs x goal_feats

    for idx, goal in zip(overshooting_idxs, chosen_ags):
      self.current_goals[idx] = goal
      self.replaced_goal[idx] = 1.

    #checking if the goals are already achieved for debugging
    """
    computed_reward = self.env.compute_reward(experience.next_state['achieved_goal'], self.current_goals, 
      {'s':experience.state['observation'], 'a':experience.action, 'ns':experience.next_state['observation']})
    close = computed_reward > -0.5
    for i in overshooting_idxs:
      #print(i)
      print(close[i])
    """
  def _process_experience(self, experience):
    if self.config.ikde:
      self._process_experience_swap(experience)
    
    elif self.config.other_args['transport']:
      self._process_experience_transport(experience)
    else:
      self._process_experience_mega(experience)
  def _process_experience_swap(self, experience):
    """Curiosity module updates the desired goal depending on experience.trajectory_over"""
    ag_buffer = self.replay_buffer.buffer.BUFF.buffer_ag

    if self.current_goals is None:
      self.current_goals = experience.reset_state['desired_goal']

    computed_reward = self.env.compute_reward(experience.next_state['achieved_goal'], self.current_goals, 
      {'s':experience.state['observation'], 'a':experience.action, 'ns':experience.next_state['observation']})
    close = computed_reward > -0.5

    # First, manage the episode resets & any special behavior that occurs on goal achievement, like go explore / resets / overshooting
    reset_idxs, overshooting_idxs, overshooting_proposals = self._manage_resets_and_success_behaviors(experience, close)

    if reset_idxs:
      self.train.reset_next(reset_idxs)

    if overshooting_idxs and len(ag_buffer):
      self._overshoot_goals(experience, overshooting_idxs, overshooting_proposals)

    # Now consider replacing the current goals with something else:
    if np.any(experience.trajectory_over) and len(ag_buffer):
      # sample some achieved goals
      sample_idxs = np.random.randint(len(ag_buffer), size=self.num_sampled_ags * self.n_envs)
      sampled_ags = ag_buffer.get_batch(sample_idxs)
      sampled_ags = sampled_ags.reshape(self.n_envs, self.num_sampled_ags, -1)

      num_goals = 125
      d = self.goaldim
      first = sampled_ags.copy()
      first[:, :, d:3*d]=0

      second = sampled_ags.copy()
      second[:, :, 0:d]=0 
      second[:, :, 2*d:3*d]=0

      third = sampled_ags.copy()
      third[:, :, 0:2*d]=0

      if not self.ag_kde_indp.ready:
        self.ag_kde_indp._optimize(force=True)

      first = first[:,:num_goals,:]
      second = second[:,:num_goals,:]
      third = third[:,:num_goals,:]
      

      for i in range(sampled_ags.shape[0]):
        first_idx = np.argsort(self.ag_kde_indp.evaluate_log_density(first[i]))
        first[i] = first[i][first_idx]
        second_idx = np.argsort(self.ag_kde_indp.evaluate_log_density(second[i]))
        second[i] = second[i][second_idx]
        third_idx = np.argsort(self.ag_kde_indp.evaluate_log_density(third[i]))
        third[i] = third[i][third_idx]
      
      if len(self.successes) > 50:
        intrinsic_success_p = np.mean(self.successes)
        if intrinsic_success_p> 0.7:
          self.swap+=0.25
      
      if self.swap in self.swapcheck:
        self.cutoff = max(self.config.initial_cutoff, self.min_min_cutoff)
        self.swapcheck.remove(self.swap)

      if self.swap >= 3:
        top =5
        fidx = [i for i in range(top) for _ in range(top**2)]
        sidx = [i for i in range(top) for _ in range(top)]*top
        tidx = [i for i in range(top)] * (top**2)
        sampled_ags = sampled_ags[:,:125,:]
        oldgoal = copy.deepcopy(sampled_ags)
        newf = sampled_ags.copy()
        news = sampled_ags.copy()
        newt = sampled_ags.copy()
        for i in range(sampled_ags.shape[0]):
          newf[i] = first[i][fidx]
          news[i] = second[i][sidx]
          newt[i] = third[i][tidx]

        sampled_ags[:,:,0:d] = newf[:,:,0:d]
        sampled_ags[:,:,d:2*d] = news[:,:,d:2*d]
        sampled_ags[:,:,2*d:3*d] = newt[:,:,2*d:3*d]
        sampled_ags = np.concatenate((sampled_ags, oldgoal), axis = 1)
      elif self.swap >= 2:
        top =5
        fidx = [i for i in range(top) for _ in range(top**2)]
        sidx = [i for i in range(top) for _ in range(top)]*top
        tidx = [i for i in range(top)] * (top**2)

        newf = sampled_ags.copy()
        news = sampled_ags.copy()
        newt = sampled_ags.copy()
        for i in range(sampled_ags.shape[0]):
          newf[i] = first[i][fidx]
          news[i] = second[i][sidx]
          newt[i] = third[i][tidx]
        rad = np.random.randint(3)
        if rad == 0:
          sampled_ags[:,:,0:d] = newf[:,:,0:d]
        sampled_ags[:,:,d:2*d] = news[:,:,d:2*d]
        if rad ==1:
          sampled_ags[:,:,0:d] = newf[:,:,0:d]
          sampled_ags[:,:,2*d:3*d] = newt[:,:,2*d:3*d]
        if rad ==2:
          sampled_ags[:,:,d:2*d] = news[:,:,d:2*d]
          sampled_ags[:,:,2*d:3*d] = newt[:,:,2*d:3*d]
        
      elif self.swap >= 1:
        top =5
        fidx = [i for i in range(top) for _ in range(top**2)]
        sidx = [i for i in range(top) for _ in range(top)]*top
        tidx = [i for i in range(top)] * (top**2)

        newf = sampled_ags.copy()
        news = sampled_ags.copy()
        newt = sampled_ags.copy()
        for i in range(sampled_ags.shape[0]):
          newf[i] = first[i][fidx]
          news[i] = second[i][sidx]
          newt[i] = third[i][tidx]
        rad = np.random.randint(3)
        if rad == 0:
          sampled_ags[:,:,0:d] = newf[:,:,0:d]   
        if rad ==1:
          sampled_ags[:,:,d:2*d] = news[:,:,d:2*d]
        if rad ==2:
          sampled_ags[:,:,2*d:3*d] = newt[:,:,2*d:3*d]

      # compute the q-values of both the sampled achieved goals and the current goals
      states = np.tile(experience.reset_state['observation'][:, None, :], (1, self.num_sampled_ags, 1))
      states = np.concatenate((states, sampled_ags), -1).reshape(self.num_sampled_ags * self.n_envs, -1)
      states_curr = np.concatenate((experience.reset_state['observation'], self.current_goals), -1)
      states_cat = np.concatenate((states, states_curr), 0)
      
      bad_q_idxs, q_values = [], None
      if self.use_qcutoff:
        if hasattr(self, 'state_normalizer'):
          states_cat = self.state_normalizer(states_cat)
        q_values = self.compute_q(states_cat)
        q_values, curr_q = np.split(q_values, [self.num_sampled_ags * self.n_envs])
        q_values = q_values.reshape(self.n_envs, self.num_sampled_ags)

        # Set cutoff dynamically by using intrinsic_success_percent
        if len(self.successes_deque) == 10:
          self.min_cutoff = max(self.min_min_cutoff, min(np.min(q_values), self.min_cutoff))
          intrinsic_success_percent = np.mean(self.successes_deque)
          if intrinsic_success_percent >= self.config.cutoff_success_threshold[1]:
            self.cutoff = max(self.min_cutoff, self.cutoff - 1.)
            self.successes_deque.clear()
          elif intrinsic_success_percent <= self.config.cutoff_success_threshold[0]:
            self.cutoff = max(min(self.config.initial_cutoff, self.cutoff + 1.), self.min_min_cutoff)
            self.successes_deque.clear()
        # zero out the "bad" values. This practically eliminates them as candidates if any goals are viable.
        bad_q_idxs = q_values < self.cutoff
        ###### show the amount of goals that was cutoff
  
        print(str(np.count_nonzero(bad_q_idxs))+'==cutoff:' + str(self.cutoff)+'==swap'+str(self.swap))
        

        #print(q_values)
        q_values[bad_q_idxs] *= -1
        min_q_values = np.min(q_values, axis=1, keepdims=True)  # num_envs x1
        q_values[bad_q_idxs] *= -1

      # score the goals -- lower is better
      goal_values = self.score_goals(sampled_ags, AttrDict(q_values=q_values, states=states))

      if self.config.dg_score_multiplier > 1. and self.dg_kde.ready:
        dg_scores = self.dg_kde.evaluate_log_density(sampled_ags.reshape(self.n_envs * self.num_sampled_ags, -1))
        dg_scores = dg_scores.reshape(self.n_envs, self.num_sampled_ags)
        goal_values[dg_scores > -np.inf] *= self.config.dg_score_multiplier

      if q_values is not None:
        goal_values[bad_q_idxs] = q_values[bad_q_idxs] * -1e-8

      if self.randomize:  # sample proportional to the absolute score
        abs_goal_values = np.abs(goal_values)
        normalized_values = abs_goal_values / np.sum(abs_goal_values, axis=1, keepdims=True)
        chosen_idx = (normalized_values.cumsum(1) > np.random.rand(normalized_values.shape[0])[:, None]).argmax(1)
      else:  # take minimum
        chosen_idx = np.argmin(goal_values, axis=1)

      chosen_idx = np.eye(self.num_sampled_ags)[chosen_idx]  # shape(sampled_ags) = n_envs x num_sampled_ags
      if q_values is not None:
        chosen_q_val = (chosen_idx * q_values).sum(axis=1, keepdims=True)
      chosen_ags = np.sum(sampled_ags * chosen_idx[:, :, None], axis=1)  # n_envs x goal_feats
      
      # replace goal always when first_visit_succ (relying on the dg_score_multiplier to dg focus), otherwise
      # we are going to transition into the dgs using the ag_kde_tophat
      if hasattr(self, 'curiosity_alpha'):
        if self.use_qcutoff:
          replace_goal = np.logical_or((np.random.random((self.n_envs, 1)) > self.curiosity_alpha.alpha),
                                       curr_q < self.cutoff).astype(np.float32)
        else:
          replace_goal = (np.random.random((self.n_envs, 1)) > self.curiosity_alpha.alpha).astype(np.float32)

      else:
        replace_goal = np.ones((self.n_envs, 1), dtype=np.float32)

      # sometimes keep the desired goal anyways
      replace_goal *= (np.random.uniform(size=[self.n_envs, 1]) > self.keep_dg_percent).astype(np.float32)

      new_goals = replace_goal * chosen_ags + (1 - replace_goal) * self.current_goals

      if hasattr(self, 'logger') and len(self.successes) > 50:
        if q_values is not None:
          self.logger.add_histogram('Explore/Goal_q', replace_goal * chosen_q_val + (1 - replace_goal) * curr_q)
        self.logger.add_scalar('Explore/Intrinsic_success_percent', np.mean(self.successes))
        self.logger.add_scalar('Explore/Cutoff', self.cutoff)
        self.successes = []

      replace_goal = replace_goal.reshape(-1)

      for i in range(self.n_envs):
        if experience.trajectory_over[i]:
          self.successes.append(float(self.is_success[i, 0] >= 1.))  # compromise due to exploration
          self.successes_deque.append(float(self.is_success[i, 0] >= 1.))  # compromise due to exploration
          self.current_goals[i] = new_goals[i]
          if replace_goal[i]:
            self.replaced_goal[i] = 1.
          self.go_explore[i] = 0.
          self.is_success[i] = 0.
      self.step = None


  def _process_experience_mega(self, experience):
    """Curiosity module updates the desired goal depending on experience.trajectory_over"""
    ag_buffer = self.replay_buffer.buffer.BUFF.buffer_ag

    if self.current_goals is None:
      self.current_goals = experience.reset_state['desired_goal']

    computed_reward = self.env.compute_reward(experience.next_state['achieved_goal'], self.current_goals, 
      {'s':experience.state['observation'], 'a':experience.action, 'ns':experience.next_state['observation']})
    close = computed_reward > -0.5

    # First, manage the episode resets & any special behavior that occurs on goal achievement, like go explore / resets / overshooting
    reset_idxs, overshooting_idxs, overshooting_proposals = self._manage_resets_and_success_behaviors(experience, close)

    if reset_idxs:
      self.train.reset_next(reset_idxs)

    if overshooting_idxs and len(ag_buffer):
      self._overshoot_goals(experience, overshooting_idxs, overshooting_proposals)

    # Now consider replacing the current goals with something else:
    if np.any(experience.trajectory_over) and len(ag_buffer):
      # sample some achieved goals
      sample_idxs = np.random.randint(len(ag_buffer), size=self.num_sampled_ags * self.n_envs)
      sampled_ags = ag_buffer.get_batch(sample_idxs)
      sampled_ags = sampled_ags.reshape(self.n_envs, self.num_sampled_ags, -1)
      
      # compute the q-values of both the sampled achieved goals and the current goals
      states = np.tile(experience.reset_state['observation'][:, None, :], (1, self.num_sampled_ags, 1))
      states = np.concatenate((states, sampled_ags), -1).reshape(self.num_sampled_ags * self.n_envs, -1)
      states_curr = np.concatenate((experience.reset_state['observation'], self.current_goals), -1)
      states_cat = np.concatenate((states, states_curr), 0)

      bad_q_idxs, q_values = [], None
      if self.use_qcutoff:
        
        if hasattr(self, 'state_normalizer'):
          states_cat = self.state_normalizer(states_cat)
        q_values = self.compute_q(states_cat)
        q_values, curr_q = np.split(q_values, [self.num_sampled_ags * self.n_envs])
        q_values = q_values.reshape(self.n_envs, self.num_sampled_ags)

        # Set cutoff dynamically by using intrinsic_success_percent
        if len(self.successes_deque) == 10:
          self.min_cutoff = max(self.min_min_cutoff, min(np.min(q_values), self.min_cutoff))
          intrinsic_success_percent = np.mean(self.successes_deque)
          if intrinsic_success_percent >= self.config.cutoff_success_threshold[1]:
            self.cutoff = max(self.min_cutoff, self.cutoff - 1.)
            self.successes_deque.clear()
          elif intrinsic_success_percent <= self.config.cutoff_success_threshold[0]:
            self.cutoff = max(min(self.config.initial_cutoff, self.cutoff + 1.), self.min_min_cutoff)
            self.successes_deque.clear()

        # zero out the "bad" values. This practically eliminates them as candidates if any goals are viable.
        bad_q_idxs = q_values < self.cutoff
        q_values[bad_q_idxs] *= -1
        min_q_values = np.min(q_values, axis=1, keepdims=True)  # num_envs x1
        q_values[bad_q_idxs] *= -1

      # score the goals -- lower is better
      goal_values = self.score_goals(sampled_ags, AttrDict(q_values=q_values, states=states))

      if self.config.dg_score_multiplier > 1. and self.dg_kde.ready:
        dg_scores = self.dg_kde.evaluate_log_density(sampled_ags.reshape(self.n_envs * self.num_sampled_ags, -1))
        dg_scores = dg_scores.reshape(self.n_envs, self.num_sampled_ags)
        goal_values[dg_scores > -np.inf] *= self.config.dg_score_multiplier

      if q_values is not None:
        goal_values[bad_q_idxs] = q_values[bad_q_idxs] * -1e-8

      if self.randomize:  # sample proportional to the absolute score
        abs_goal_values = np.abs(goal_values)
        normalized_values = abs_goal_values / np.sum(abs_goal_values, axis=1, keepdims=True)
        chosen_idx = (normalized_values.cumsum(1) > np.random.rand(normalized_values.shape[0])[:, None]).argmax(1)
      else:  # take minimum
        chosen_idx = np.argmin(goal_values, axis=1)

      chosen_idx = np.eye(self.num_sampled_ags)[chosen_idx]  # shape(sampled_ags) = n_envs x num_sampled_ags
      if q_values is not None:
        chosen_q_val = (chosen_idx * q_values).sum(axis=1, keepdims=True)
      chosen_ags = np.sum(sampled_ags * chosen_idx[:, :, None], axis=1)  # n_envs x goal_feats

      # replace goal always when first_visit_succ (relying on the dg_score_multiplier to dg focus), otherwise
      # we are going to transition into the dgs using the ag_kde_tophat
      if hasattr(self, 'curiosity_alpha'):
        if self.use_qcutoff:
          replace_goal = np.logical_or((np.random.random((self.n_envs, 1)) > self.curiosity_alpha.alpha),
                                       curr_q < self.cutoff).astype(np.float32)
        else:
          replace_goal = (np.random.random((self.n_envs, 1)) > self.curiosity_alpha.alpha).astype(np.float32)

      else:
        replace_goal = np.ones((self.n_envs, 1), dtype=np.float32)

      # sometimes keep the desired goal anyways
      replace_goal *= (np.random.uniform(size=[self.n_envs, 1]) > self.keep_dg_percent).astype(np.float32)

      new_goals = replace_goal * chosen_ags + (1 - replace_goal) * self.current_goals
      
      if hasattr(self, 'logger') and len(self.successes) > 50:
        

        if q_values is not None:
          self.logger.add_histogram('Explore/Goal_q', replace_goal * chosen_q_val + (1 - replace_goal) * curr_q)
        self.logger.add_scalar('Explore/Intrinsic_success_percent', np.mean(self.successes))
        self.logger.add_scalar('Explore/Cutoff', self.cutoff)
        self.successes = []

      replace_goal = replace_goal.reshape(-1)

      for i in range(self.n_envs):
        if experience.trajectory_over[i]:
          self.successes.append(float(self.is_success[i, 0] >= 1.))  # compromise due to exploration
          self.successes_deque.append(float(self.is_success[i, 0] >= 1.))  # compromise due to exploration
          self.current_goals[i] = new_goals[i]
          if replace_goal[i]:
            self.replaced_goal[i] = 1.
          self.go_explore[i] = 0.
          self.is_success[i] = 0.
      self.step = None

  def _process_experience_transport(self, experience):
    """Curiosity module updates the desired goal depending on experience.trajectory_over and uses EMD transport"""
    ag_buffer = self.replay_buffer.buffer.BUFF.buffer_ag

    if self.current_goals is None:
      self.current_goals = experience.reset_state['desired_goal']

    computed_reward = self.env.compute_reward(experience.next_state['achieved_goal'], self.current_goals, 
      {'s':experience.state['observation'], 'a':experience.action, 'ns':experience.next_state['observation']})
    close = computed_reward > -0.5

    # First, manage the episode resets & any special behavior that occurs on goal achievement, like go explore / resets / overshooting
    reset_idxs, overshooting_idxs, overshooting_proposals = self._manage_resets_and_success_behaviors(experience, close)

    if reset_idxs:
      self.train.reset_next(reset_idxs)

    if overshooting_idxs and len(ag_buffer):
      self._overshoot_goals(experience, overshooting_idxs, overshooting_proposals)

    # Now consider replacing the current goals with something else:
    if np.any(experience.trajectory_over) and len(ag_buffer):
      # sample some achieved goals
      sample_idxs = np.random.randint(len(ag_buffer), size=self.num_sampled_ags * self.n_envs)
      sampled_ags = ag_buffer.get_batch(sample_idxs)
      sampled_ags = sampled_ags.reshape(self.n_envs, self.num_sampled_ags, -1)

      # compute the q-values of both the sampled achieved goals and the current goals
      states = np.tile(experience.reset_state['observation'][:, None, :], (1, self.num_sampled_ags, 1))
      states = np.concatenate((states, sampled_ags), -1).reshape(self.num_sampled_ags * self.n_envs, -1)
      states_curr = np.concatenate((experience.reset_state['observation'], self.current_goals), -1)
      states_cat = np.concatenate((states, states_curr), 0)
      
      bad_q_idxs, q_values = [], None
      if self.use_qcutoff:
        q_values = self.compute_q(states_cat)
        q_values, curr_q = np.split(q_values, [self.num_sampled_ags * self.n_envs])
        q_values = q_values.reshape(self.n_envs, self.num_sampled_ags)

        # Set cutoff dynamically by using intrinsic_success_percent
        if len(self.successes_deque) == 10:
          self.min_cutoff = max(self.min_min_cutoff, min(np.min(q_values), self.min_cutoff))
          intrinsic_success_percent = np.mean(self.successes_deque)
          if intrinsic_success_percent >= self.config.cutoff_success_threshold[1]:
            self.cutoff = max(self.min_cutoff, self.cutoff - 1.)
            self.successes_deque.clear()
          elif intrinsic_success_percent <= self.config.cutoff_success_threshold[0]:
            self.cutoff = max(min(self.config.initial_cutoff, self.cutoff + 1.), self.min_min_cutoff)
            self.successes_deque.clear()

        # zero out the "bad" values. This practically eliminates them as candidates if any goals are viable.
        bad_q_idxs = q_values < self.cutoff
        q_values[bad_q_idxs] *= -1
        min_q_values = np.min(q_values, axis=1, keepdims=True)  # num_envs x1
        q_values[bad_q_idxs] *= -1

      # score the goals -- lower is better
      goal_values = self.score_goals(sampled_ags, AttrDict(q_values=q_values, states=states))
      
      #save top 10% goals
      print(self.past_goals)
      if len(self.past_goals) < 1:
        print('here')
        # save top 10% of the goals
        save = copy.deepcopy(sampled_ags)
        top = sampled_ags.shape[0]//10
        top_idx = np.argpartition(goal_values, top, axis=1) 
        for i in range(sampled_ags.shape[0]):
          save[i] = save[i][top_idx[i]]

        
        top_goals = save[:,top_idx]
        top_goals = top_goals[:,:top,:]
        self.past_goals.append(top_goals)
          
      else:
        #save and then transport
        save = copy.deepcopy(sampled_ags)
        top = sampled_ags.shape[0]//10
        top_idx = np.argpartition(goal_values, top, axis=1) 
        for i in range(sampled_ags.shape[0]):
          save[i] = save[i][top_idx[i]]

        
        top_goals = save[:,top_idx]
        top_goals = top_goals[:,:top,:]
        self.past_goals.append(top_goals)
        past_goals = self.past_goals.pop(0)
        M = ot.dist(top_goals, past_goals)
        M /= M.max()
        tp = ot.emd([], [], M)
        tp = tp < 0.3
        print(tp)
        tp.nonzero()
        

        
      
      
      # compute the q-values of both the sampled achieved goals and the current goals
      states = np.tile(experience.reset_state['observation'][:, None, :], (1, self.num_sampled_ags, 1))
      states = np.concatenate((states, sampled_ags), -1).reshape(self.num_sampled_ags * self.n_envs, -1)
      states_curr = np.concatenate((experience.reset_state['observation'], self.current_goals), -1)
      states_cat = np.concatenate((states, states_curr), 0)
      
      bad_q_idxs, q_values = [], None
      if self.use_qcutoff:
        q_values = self.compute_q(states_cat)
        q_values, curr_q = np.split(q_values, [self.num_sampled_ags * self.n_envs])
        q_values = q_values.reshape(self.n_envs, self.num_sampled_ags)

        # Set cutoff dynamically by using intrinsic_success_percent
        if len(self.successes_deque) == 10:
          self.min_cutoff = max(self.min_min_cutoff, min(np.min(q_values), self.min_cutoff))
          intrinsic_success_percent = np.mean(self.successes_deque)
          if intrinsic_success_percent >= self.config.cutoff_success_threshold[1]:
            self.cutoff = max(self.min_cutoff, self.cutoff - 1.)
            self.successes_deque.clear()
          elif intrinsic_success_percent <= self.config.cutoff_success_threshold[0]:
            self.cutoff = max(min(self.config.initial_cutoff, self.cutoff + 1.), self.min_min_cutoff)
            self.successes_deque.clear()

        # zero out the "bad" values. This practically eliminates them as candidates if any goals are viable.
        bad_q_idxs = q_values < self.cutoff
        q_values[bad_q_idxs] *= -1
        min_q_values = np.min(q_values, axis=1, keepdims=True)  # num_envs x1
        q_values[bad_q_idxs] *= -1

      # score the goals -- lower is better
      goal_values = self.score_goals(sampled_ags, AttrDict(q_values=q_values, states=states))
      

      if self.config.dg_score_multiplier > 1. and self.dg_kde.ready:
        dg_scores = self.dg_kde.evaluate_log_density(sampled_ags.reshape(self.n_envs * self.num_sampled_ags, -1))
        dg_scores = dg_scores.reshape(self.n_envs, self.num_sampled_ags)
        goal_values[dg_scores > -np.inf] *= self.config.dg_score_multiplier

      if q_values is not None:
        goal_values[bad_q_idxs] = q_values[bad_q_idxs] * -1e-8

      if self.randomize:  # sample proportional to the absolute score
        abs_goal_values = np.abs(goal_values)
        normalized_values = abs_goal_values / np.sum(abs_goal_values, axis=1, keepdims=True)
        chosen_idx = (normalized_values.cumsum(1) > np.random.rand(normalized_values.shape[0])[:, None]).argmax(1)
      else:  # take minimum
        chosen_idx = np.argmin(goal_values, axis=1)

      chosen_idx = np.eye(self.num_sampled_ags)[chosen_idx]  # shape(sampled_ags) = n_envs x num_sampled_ags
      if q_values is not None:
        chosen_q_val = (chosen_idx * q_values).sum(axis=1, keepdims=True)
      chosen_ags = np.sum(sampled_ags * chosen_idx[:, :, None], axis=1)  # n_envs x goal_feats

      # replace goal always when first_visit_succ (relying on the dg_score_multiplier to dg focus), otherwise
      # we are going to transition into the dgs using the ag_kde_tophat
      if hasattr(self, 'curiosity_alpha'):
        if self.use_qcutoff:
          replace_goal = np.logical_or((np.random.random((self.n_envs, 1)) > self.curiosity_alpha.alpha),
                                       curr_q < self.cutoff).astype(np.float32)
        else:
          replace_goal = (np.random.random((self.n_envs, 1)) > self.curiosity_alpha.alpha).astype(np.float32)

      else:
        replace_goal = np.ones((self.n_envs, 1), dtype=np.float32)

      # sometimes keep the desired goal anyways
      replace_goal *= (np.random.uniform(size=[self.n_envs, 1]) > self.keep_dg_percent).astype(np.float32)

      new_goals = replace_goal * chosen_ags + (1 - replace_goal) * self.current_goals
      
      if hasattr(self, 'logger') and len(self.successes) > 50:
        
        if q_values is not None:
          self.logger.add_histogram('Explore/Goal_q', replace_goal * chosen_q_val + (1 - replace_goal) * curr_q)
        self.logger.add_scalar('Explore/Intrinsic_success_percent', np.mean(self.successes))
        self.logger.add_scalar('Explore/Cutoff', self.cutoff)
        self.successes = []

      replace_goal = replace_goal.reshape(-1)

      for i in range(self.n_envs):
        if experience.trajectory_over[i]:
          self.successes.append(float(self.is_success[i, 0] >= 1.))  # compromise due to exploration
          self.successes_deque.append(float(self.is_success[i, 0] >= 1.))  # compromise due to exploration
          self.current_goals[i] = new_goals[i]
          if replace_goal[i]:
            self.replaced_goal[i] = 1.
          self.go_explore[i] = 0.
          self.is_success[i] = 0.
      self.step = None

  def compute_q(self, numpy_states):
    states = self.torch(numpy_states)
    max_actions = self.actor(states)
    if isinstance(max_actions, tuple):
      max_actions = max_actions[0]
    return self.numpy(self.critic(states, max_actions))

  def relabel_state(self, state):
    """Should be called by the policy module to relabel states with intrinsic goals"""
    if self.current_goals is None:
      return state

    return {
        'observation': state['observation'],
        'achieved_goal': state['achieved_goal'],
        'desired_goal': self.current_goals
    }

  def score_goals(self, sampled_ags, info):
    """ Lower is better """
    raise NotImplementedError  # SUBCLASS THIS!

  def save(self, save_folder):
    self._save_props(['cutoff', 'min_cutoff'], save_folder)  #can restart keeping track of successes / go explore

  def load(self, save_folder):
    self._load_props(['cutoff', 'min_cutoff'], save_folder)


class QAchievedGoalCuriosity(AchievedGoalCuriosity):
  """
  Scores goals by the Q values (lower is better)
  """
  def score_goals(self, sampled_ags, info):
    scores = np.copy(info.q_values)
    max_score = np.max(scores)
    if max_score > 0:
      scores -= max_score  # so all scores negative

    return scores


class SuccessAchievedGoalCuriosity(AchievedGoalCuriosity):
  """
  Scores goals based on success prediction by a goal discriminator module.
  """
  def _setup(self):
    super()._setup()
    self.use_qcutoff = False

  def score_goals(self, sampled_ags, info):

    # sampled_ags is np.array of shape NUM_ENVS x NUM_SAMPLED_GOALS (both arbitrary)
    num_envs, num_sampled_ags = sampled_ags.shape[:2]

    scores = self.success_predictor(info.states).reshape(num_envs, num_sampled_ags)  # these are predicted success %
    scores = -0.5 + np.abs(scores - 0.5)  # rank by distance to 0.5, lower is closer to 0.5

    return scores


class DensityAchievedGoalCuriosity(AchievedGoalCuriosity):
  """
  Scores goals by their densities (lower is better), using KDE to estimate

  Note on bandwidth: it seems bandwith = 0.1 works pretty well with normalized samples (which is
  why we normalize the ags).
  """
  def __init__(self, density_module='ag_kde', interest_module='ag_interest', alpha=-1.0, **kwargs):
    super().__init__(**kwargs)
    self.alpha = alpha
    self.density_module = density_module
    self.interest_module = interest_module

  def _setup(self):
    assert hasattr(self, self.density_module)
    super()._setup()

  def score_goals(self, sampled_ags, info):
    """ Lower is better """
    density_module = getattr(self, self.density_module)
    if not density_module.ready:
      density_module._optimize(force=True)
    interest_module = None
    if hasattr(self, self.interest_module):
      interest_module = getattr(self, self.interest_module)
      if not interest_module.ready:
        interest_module = None

    # sampled_ags is np.array of shape NUM_ENVS x NUM_SAMPLED_GOALS (both arbitrary)
    num_envs, num_sampled_ags = sampled_ags.shape[:2]

    # score the sampled_ags to get log densities, and exponentiate to get densities
    flattened_sampled_ags = sampled_ags.reshape(num_envs * num_sampled_ags, -1)
    sampled_ag_scores = density_module.evaluate_log_density(flattened_sampled_ags)
    if interest_module:
      # Interest is ~(det(feature_transform)), so we subtract it  in order to add ~(det(inverse feature_transform)) for COV.
      sampled_ag_scores -= interest_module.evaluate_log_interest(flattened_sampled_ags)  # add in log interest
    sampled_ag_scores = sampled_ag_scores.reshape(num_envs, num_sampled_ags)  # these are log densities

    # Take softmax of the alpha * log density.
    # If alpha = -1, this gives us normalized inverse densities (higher is rarer)
    # If alpha < -1, this skews the density to give us low density samples
    normalized_inverse_densities = softmax(sampled_ag_scores * self.alpha)
    normalized_inverse_densities *= -1.  # make negative / reverse order so that lower is better.

    return normalized_inverse_densities


class EntropyGainScoringGoalCuriosity(AchievedGoalCuriosity):
  """
  Scores goals by their expected entropy gain (higher is better), using KDE to estimate
  current density and another KDE to estimate the joint likelihood of achieved goal 
  given behavioural goal.
  """
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def _setup(self):
    assert hasattr(self, 'bg_kde')
    assert hasattr(self, 'ag_kde')
    assert hasattr(self, 'bgag_kde')
    assert hasattr(self, 'replay_buffer')
    super()._setup()

  def score_goals(self, sampled_ags, info):
    """ Higher entropy gain is better """
    if not self.ag_kde.ready:
      self.ag_kde._optimize(force=True)

    if not self.bg_kde.ready:
      self.bg_kde._optimize(force=True)

    if not self.bgag_kde.ready:
      self.bgag_kde._optimize(force=True)

    # sampled_ags is np.array of shape NUM_ENVS x NUM_SAMPLED_GOALS (both arbitrary)
    num_envs, num_sampled_ags = sampled_ags.shape[:2]

    # Get sample of predicted achieved goal from mixture density network
    candidate_bgs = sampled_ags.reshape(num_envs * num_sampled_ags, -1)

    # Reuse the candidate bgs as potential ags
    # Note: We are using a sliding window to reuse sampled_ags as the potential ag for each bg
    # Prior that each bgs has one ag that is identical to bg, i.e. that it reaches the bg.
    num_ags = 10  # TODO: Not make it hard coded
    indexer = np.arange(num_envs * num_sampled_ags).reshape(-1, 1) + np.arange(num_ags).reshape(1, -1)
    indexer %= num_envs * num_sampled_ags  # To wrap around to the beginning
    ags_samples = np.concatenate(
        [candidate_bgs[indexer[i]][np.newaxis, :, :] for i in range(num_envs * num_sampled_ags)], axis=0)

    candidate_bgs_repeat = np.repeat(candidate_bgs[:, np.newaxis, :], num_ags,
                                     axis=1)  # Shape num_envs*num_sampled_ags, num_ags, dim
    joint_candidate_bgags = np.concatenate([candidate_bgs_repeat, ags_samples], axis=-1)
    joint_candidate_bgags = joint_candidate_bgags.reshape(num_envs * num_sampled_ags * num_ags, -1)

    # score the sampled_ags to get log densities, and exponentiate to get densities
    joint_candidate_score = self.bgag_kde.evaluate_log_density(joint_candidate_bgags)
    joint_candidate_score = joint_candidate_score.reshape(num_envs * num_sampled_ags,
                                                          num_ags)  # these are log densities

    candidate_bgs_score = self.bg_kde.evaluate_log_density(
        candidate_bgs_repeat.reshape(num_envs * num_sampled_ags * num_ags, -1))
    candidate_bgs_score = candidate_bgs_score.reshape(num_envs * num_sampled_ags, num_ags)  # these are log densities
    cond_candidate_score = joint_candidate_score - candidate_bgs_score
    cond_candidate_score = softmax(cond_candidate_score, axis=1)

    # Compute entropy gain for the predicted achieved goal
    beta = 1 / len(self.replay_buffer.buffer)
    sampled_ag_entr_new = self.ag_kde.evaluate_elementwise_entropy(candidate_bgs, beta=beta)
    sampled_ag_entr_old = self.ag_kde.evaluate_elementwise_entropy(candidate_bgs, beta=0.)
    sampled_ag_entr_gain = sampled_ag_entr_new - sampled_ag_entr_old
    sampled_ag_entr_gain /= beta  # Normalize by beta # TODO: Get rid of this part if not necessary
    sampled_ag_entr_gain = np.concatenate(
        [sampled_ag_entr_gain[indexer[i]][np.newaxis, :] for i in range(num_envs * num_sampled_ags)], axis=0)
    sampled_ag_entr_gain *= cond_candidate_score
    sampled_ag_entr_gain = sampled_ag_entr_gain.mean(axis=1)

    scores = sampled_ag_entr_gain.reshape(num_envs, num_sampled_ags)
    scores *= -1.  # make negative / reverse order so that lower is better.

    return scores


class CuriosityAlphaMixtureModule(mrl.Module):
  """
    For curiosity agents; this module approximates alpha = (1 / (1 + KL)) using the ag_kde density estimator for p_ag.
  """
  def __init__(self, optimize_every=100):
    super().__init__('curiosity_alpha',
                     required_agent_modules=['ag_curiosity', 'ag_kde', 'replay_buffer'],
                     locals=locals())
    self.samples = None
    self.bandwidth = None
    self.kernel = None
    self.kde = None
    self.fitted_kde = None
    self._alpha = 0.
    self._beta = -3.
    self.step = 0
    self.optimize_every = optimize_every

  def _setup(self):
    self.samples = self.ag_kde.samples
    self.bandwidth = self.ag_kde.bandwidth
    self.kernel = self.ag_kde.kernel
    self.kde = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth)
    if 'curiosity_beta' in self.config:
      self._beta = self.config.curiosity_beta

  @property
  def alpha(self):
    return self._alpha

  def _optimize(self):
    buffer = self.replay_buffer.buffer.BUFF['buffer_dg']
    self.step += 1

    if self.step % self.optimize_every == 0 and len(buffer):

      # Fit the DG KDE
      num_samples = 1000
      sample_idxs = np.random.randint(len(buffer), size=num_samples)
      kde_samples = buffer.get_batch(sample_idxs)
      kde_samples = (kde_samples - self.ag_kde.kde_sample_mean) / self.ag_kde.kde_sample_std
      self.fitted_kde = self.kde.fit(kde_samples)

      # Now compute alpha
      s = kde_samples
      log_p_dg = self.fitted_kde.score_samples(s)
      log_p_ag = self.ag_kde.fitted_kde.score_samples(s)
      self._alpha = 1. / max((self._beta + np.mean(log_p_dg) - np.mean(log_p_ag)), 1.)

      # Occasionally log the alpha
      self.logger.add_scalar('Explore/curiosity_alpha', self._alpha, log_every=500)
      self.logger.add_tabular('Curiosity_alpha', self._alpha)

  def save(self, save_folder):
    self._save_props(['kde', 'samples', 'bandwidth', 'kernel'], save_folder)

  def load(self, save_folder):
    self._load_props(['kde', 'samples', 'bandwidth', 'kernel'], save_folder)
