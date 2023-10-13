import mrl
from mrl.utils.misc import AttrDict
import numpy as np
import csv

class EpisodicEval(mrl.Module):
  def __init__(self):
    super().__init__('eval', required_agent_modules = ['eval_env', 'policy'], locals=locals())
  
  def __call__(self, num_episodes : int, plot= False, *unused_args):
    """
    Runs num_steps steps in the environment and returns results.
    Results tracking is done here instead of in process_experience, since 
    experiences aren't "real" experiences; e.g. agent cannot learn from them.  
    """
    self.eval_mode()
    env = self.eval_env
    num_envs = env.num_envs
    
    episode_rewards, episode_steps = [], []
    discounted_episode_rewards = []
    is_successes = []
    record_success = False
    f = open('plot.csv', 'a+')
    writer = csv.writer(f)
    while len(episode_rewards) < num_episodes:
      state = env.reset()
      qstate = np.concatenate((state['observation'], state['desired_goal']), -1)
      obs = state['observation']
      dgoal = state['desired_goal']
      dones = np.zeros((num_envs,))
      steps = np.zeros((num_envs,))
      is_success = np.zeros((num_envs,))
      ep_rewards = [[] for _ in range(num_envs)]

      while not np.all(dones):
        action = self.policy(state)
        state, reward, dones_, infos = env.step(action)


        for i, (rew, done, info) in enumerate(zip(reward, dones_, infos)):
          if dones[i]:
            continue
          ep_rewards[i].append(rew)
          steps[i] += 1
          if done:
            dones[i] = 1. 
            if 'is_success' in info:
              record_success = True
              is_success[i] = info['is_success']
      
      if hasattr(self, 'state_normalizer'):
        qstate = self.state_normalizer(qstate)
        
      q_values = self.ag_curiosity.compute_q(qstate)

      ags = state['desired_goal']
      #print(ags)
      nobuf= False
      density_module = self.agent.ag_kde
      if not density_module.ready:
        buffer = getattr(density_module, density_module.buffer_name).buffer.BUFF['buffer_' + density_module.item]
        if len(buffer)==0:
          nobuf= True
        else:
          density_module._optimize(force=True)
      num_envs = 1
      num_sampled_ags = ags.shape[0]
      predictions = self.success_predictor(obs,dgoal).reshape(num_envs, num_sampled_ags) 
      # score the sampled_ags to get log densities, and exponentiate to get densities
      flattened_sampled_ags = ags.reshape(num_envs * num_sampled_ags, -1)
      if not nobuf:
        sampled_ag_scores = density_module.evaluate_log_density(flattened_sampled_ags)
        
      else:
        sampled_ag_scores= [-1]
      #row = [str(i) for i in is_success] + ['score'] + [str(i) for i in q_values]
      #writer.writerow(row)
      
      i=0
      for ep_reward, step, is_succ in zip(ep_rewards, steps, is_success):
        if record_success:
          is_successes.append(is_succ)
        episode_rewards.append(sum(ep_reward))
        discounted_episode_rewards.append(discounted_sum(ep_reward, self.config.gamma))
        episode_steps.append(step)
        #row = [str(discounted_sum(ep_reward, self.config.gamma))]+ ['score'] + [str(predictions[i])]
        row = [str(i) for i in is_success]+ ['score'] + [str(predictions[i])]
        i +=1
        writer.writerow(row)
      
    
    
    f.close()
    if hasattr(self, 'logger'):
      if len(is_successes):
        self.logger.add_scalar('Test/Success', np.mean(is_successes))
      self.logger.add_scalar('Test/Episode_rewards', np.mean(episode_rewards))
      self.logger.add_scalar('Test/Discounted_episode_rewards', np.mean(discounted_episode_rewards))
      self.logger.add_scalar('Test/Episode_steps', np.mean(episode_steps))

    return AttrDict({
      'rewards': episode_rewards,
      'steps': episode_steps
    })

class EpisodicEval(mrl.Module):
  def __init__(self):
    super().__init__('eval', required_agent_modules = ['eval_env', 'policy'], locals=locals())
  
  def __call__(self, num_episodes : int, plot= False, *unused_args):
    """
    Runs num_steps steps in the environment and returns results.
    Results tracking is done here instead of in process_experience, since 
    experiences aren't "real" experiences; e.g. agent cannot learn from them.  
    """
    self.eval_mode()
    env = self.eval_env
    num_envs = env.num_envs
    
    episode_rewards, episode_steps = [], []
    discounted_episode_rewards = []
    is_successes = []
    record_success = False
    f = open('plot.csv', 'a+')
    writer = csv.writer(f)
    while len(episode_rewards) < num_episodes:
      state = env.reset()
      qstate = np.concatenate((state['observation'], state['desired_goal']), -1)
      obs = state['observation']
      dgoal = state['desired_goal']
      dones = np.zeros((num_envs,))
      steps = np.zeros((num_envs,))
      is_success = np.zeros((num_envs,))
      ep_rewards = [[] for _ in range(num_envs)]

      while not np.all(dones):
        action = self.policy(state)
        state, reward, dones_, infos = env.step(action)


        for i, (rew, done, info) in enumerate(zip(reward, dones_, infos)):
          if dones[i]:
            continue
          ep_rewards[i].append(rew)
          steps[i] += 1
          if done:
            dones[i] = 1. 
            if 'is_success' in info:
              record_success = True
              is_success[i] = info['is_success']
      
      if hasattr(self, 'state_normalizer'):
        qstate = self.state_normalizer(qstate)
        
      q_values = self.ag_curiosity.compute_q(qstate)

      ags = state['desired_goal']
      #print(ags)
      nobuf= False
      density_module = self.agent.ag_kde
      if not density_module.ready:
        buffer = getattr(density_module, density_module.buffer_name).buffer.BUFF['buffer_' + density_module.item]
        if len(buffer)==0:
          nobuf= True
        else:
          density_module._optimize(force=True)
      num_envs = 1
      num_sampled_ags = ags.shape[0]
      predictions = self.success_predictor(obs,dgoal).reshape(num_envs, num_sampled_ags) 
      # score the sampled_ags to get log densities, and exponentiate to get densities
      flattened_sampled_ags = ags.reshape(num_envs * num_sampled_ags, -1)
      if not nobuf:
        sampled_ag_scores = density_module.evaluate_log_density(flattened_sampled_ags)
        
      else:
        sampled_ag_scores= [-1]
      #row = [str(i) for i in is_success] + ['score'] + [str(i) for i in q_values]
      #writer.writerow(row)
      
      i=0
      for ep_reward, step, is_succ in zip(ep_rewards, steps, is_success):
        if record_success:
          is_successes.append(is_succ)
        episode_rewards.append(sum(ep_reward))
        discounted_episode_rewards.append(discounted_sum(ep_reward, self.config.gamma))
        episode_steps.append(step)
        #row = [str(discounted_sum(ep_reward, self.config.gamma))]+ ['score'] + [str(predictions[i])]
        row = [str(i) for i in is_success]+ ['score'] + [str(predictions[i])]
        i +=1
        writer.writerow(row)
      
    
    
    f.close()
    if hasattr(self, 'logger'):
      if len(is_successes):
        self.logger.add_scalar('Test/Success', np.mean(is_successes))
      self.logger.add_scalar('Test/Episode_rewards', np.mean(episode_rewards))
      self.logger.add_scalar('Test/Discounted_episode_rewards', np.mean(discounted_episode_rewards))
      self.logger.add_scalar('Test/Episode_steps', np.mean(episode_steps))

    return AttrDict({
      'rewards': episode_rewards,
      'steps': episode_steps
    })

def discounted_sum(lst, discount):
  sum = 0
  gamma = 1
  for i in lst:
    sum += gamma*i
    gamma *= discount
  return sum