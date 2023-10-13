# 1. Imports
from mrl.import_all import *
from mrl.modules.train import debug_vectorized_experience
from experiments.mega.make_env import make_env
import time
import os
import gym
import numpy as np
import torch.nn as nn

from mrl.replays.online_her_buffer import OnlineHERBufferValidation

# 2. Get default config and update any defaults (this automatically updates the argparse defaults)
config = protoge_config()
# config.batch_size = 2000

# 3. Make changes to the argparse below

def main(args):
  # 4. Update the config with args, and make the agent name. 
  if args.num_envs is None:
    import multiprocessing as mp
    args.num_envs = max(mp.cpu_count() - 1, 1)

  merge_args_into_config(args, config)
  
  if args.ga:
    config.gamma = 0.99

  if config.gamma < 1.: config.clip_target_range = (np.round(-(1 / (1-config.gamma)), 2), 0.)
  if config.gamma == 1: config.clip_target_range = (np.round(- args.env_max_step - 5, 2), 0.)

  if args.sparse_reward_shaping:
    config.clip_target_range = (-np.inf, np.inf)

  config.agent_name = make_agent_name(config, ['env','alg','her','layers','seed','tb','ag_curiosity','eexplore','first_visit_succ', 'dg_score_multiplier','alpha'], prefix=args.prefix)

  # 5. Setup / add basic modules to the config
  config.update(
      dict(
          trainer=StandardTrain(),
          evaluation=EpisodicEval(),
          policy=ActorPolicy(),
          logger=Logger(),
          state_normalizer=Normalizer(MeanStdNormalizer()),
          replay=OnlineHERBuffer(),
      ))

  config.prioritized_mode = args.prioritized_mode
  if config.prioritized_mode == 'mep':
    config.prioritized_replay = EntropyPrioritizedOnlineHERBuffer()
  config.ikde = args.ikde
  
  if args.ikde:
    density_module_str = 'ag_kde_indp'
  else:
    density_module_str = 'ag_kde'
  config.transport = args.transport
  
  if not args.no_ag_kde:
    config.ag_kde_indp = IndependentKernelDensity('ag', optimize_every=10, samples=10000, kernel=args.kde_kernel, bandwidth = args.bandwidth, log_entropy=True, tag='_indp')
    config.ag_kde = RawKernelDensity('ag', optimize_every=10, samples=10000, kernel=args.kde_kernel, bandwidth = args.bandwidth, log_entropy=True)
  
  if args.ag_curiosity is not None:
    config.dg_kde = RawKernelDensity('dg', optimize_every=500, samples=10000, kernel='tophat', bandwidth = 0.2)
    config.ag_kde_tophat = RawKernelDensity('ag', optimize_every=100, samples=10000, kernel='tophat', bandwidth = 0.2, tag='_tophat')
    if args.transition_to_dg:
      config.alpha_curiosity = CuriosityAlphaMixtureModule()
    if 'rnd' in args.ag_curiosity:
      config.ag_rnd = RandomNetworkDensity('ag')
    if 'flow' in args.ag_curiosity:
      config.ag_flow = FlowDensity('ag')

    use_qcutoff = not args.no_cutoff

  

    if args.ag_curiosity == 'minq':
      config.ag_curiosity = QAchievedGoalCuriosity(max_steps = args.env_max_step, num_sampled_ags=args.num_sampled_ags, use_qcutoff=use_qcutoff, keep_dg_percent=args.keep_dg_percent)
    elif args.ag_curiosity == 'randq':
      config.ag_curiosity = QAchievedGoalCuriosity(max_steps = args.env_max_step, randomize=True, num_sampled_ags=args.num_sampled_ags, use_qcutoff=use_qcutoff, keep_dg_percent=args.keep_dg_percent)
    elif args.ag_curiosity == 'minkde':
      config.ag_curiosity = DensityAchievedGoalCuriosity(density_module_str, max_steps = args.env_max_step, num_sampled_ags=args.num_sampled_ags, use_qcutoff=use_qcutoff, keep_dg_percent=args.keep_dg_percent)
      config.success_predictor = GoalSuccessPredictor(batch_size=args.succ_bs, history_length=args.succ_hl, optimize_every=args.succ_oe)
    elif args.ag_curiosity == 'minrnd':
      config.ag_curiosity = DensityAchievedGoalCuriosity('ag_rnd', max_steps = args.env_max_step, num_sampled_ags=args.num_sampled_ags, use_qcutoff=use_qcutoff, keep_dg_percent=args.keep_dg_percent)
    elif args.ag_curiosity == 'minflow':
      config.ag_curiosity = DensityAchievedGoalCuriosity('ag_flow', max_steps = args.env_max_step, num_sampled_ags=args.num_sampled_ags, use_qcutoff=use_qcutoff, keep_dg_percent=args.keep_dg_percent)
    elif args.ag_curiosity == 'randkde':
      config.ag_curiosity = DensityAchievedGoalCuriosity(alpha = args.alpha, max_steps = args.env_max_step, randomize=True, num_sampled_ags=args.num_sampled_ags, use_qcutoff=use_qcutoff, keep_dg_percent=args.keep_dg_percent)
    elif args.ag_curiosity == 'randrnd':
      config.ag_curiosity = DensityAchievedGoalCuriosity('ag_rnd', alpha = args.alpha, max_steps = args.env_max_step, num_sampled_ags=args.num_sampled_ags, use_qcutoff=use_qcutoff, keep_dg_percent=args.keep_dg_percent)
    elif args.ag_curiosity == 'randflow':
      config.ag_curiosity = DensityAchievedGoalCuriosity('ag_flow', alpha = args.alpha, max_steps = args.env_max_step, num_sampled_ags=args.num_sampled_ags, use_qcutoff=use_qcutoff, keep_dg_percent=args.keep_dg_percent)
    elif args.ag_curiosity == 'goaldisc':
      config.success_predictor = GoalSuccessPredictor(batch_size=args.succ_bs, history_length=args.succ_hl, optimize_every=args.succ_oe)
      config.ag_curiosity = SuccessAchievedGoalCuriosity(max_steps=args.env_max_step, use_qcutoff=use_qcutoff, keep_dg_percent=args.keep_dg_percent)
    elif args.ag_curiosity == 'entropygainscore':
      config.bg_kde = RawKernelDensity('bg', optimize_every=args.env_max_step, samples=10000, kernel=args.kde_kernel, bandwidth = args.bandwidth, log_entropy=True)
      config.bgag_kde = RawJointKernelDensity(['bg','ag'], optimize_every=args.env_max_step, samples=10000, kernel=args.kde_kernel, bandwidth = args.bandwidth, log_entropy=True)
      config.ag_curiosity = EntropyGainScoringGoalCuriosity(max_steps=args.env_max_step, use_qcutoff=use_qcutoff, keep_dg_percent=args.keep_dg_percent)
    else:
      raise NotImplementedError

  if args.noise_type.lower() == 'gaussian': noise_type = GaussianProcess
  if args.noise_type.lower() == 'ou': noise_type = OrnsteinUhlenbeckProcess
  config.action_noise = ContinuousActionNoise(noise_type, std=ConstantSchedule(args.action_noise))

  if args.alg.lower() == 'ddpg': 
    config.algorithm = DDPG()
  elif args.alg.lower() == 'td3':
    config.algorithm = TD3()
    config.target_network_update_freq *= 2
  elif args.alg.lower() == 'dqn': 
    config.algorithm = DQN()
    config.policy = QValuePolicy()
    config.qvalue_lr = config.critic_lr
    config.qvalue_weight_decay = config.actor_weight_decay
    config.double_q = True
    config.random_action_prob = LinearSchedule(1.0, config.eexplore, 1e5)
  elif args.alg.lower() == 'ddpg_explore': 
    config.algorithm = DDPG_EXPLORE()
    config.policy =  ExploreActorPolicy()
    config.explore_state_normalizer=Normalizer(MeanStdNormalizer())
  elif args.alg.lower() == 'clearning':
    config.algorithm = CLearning()
    config.policy =  CLearningActorPolicy()
    
  else:
    raise NotImplementedError

  # 6. Setup / add the environments and networks (which depend on the environment) to the config
  env, eval_env = make_env(args)
  if args.first_visit_done:
    env1, eval_env1 = env, eval_env
    env = lambda: FirstVisitDoneWrapper(env1())
    eval_env = lambda: FirstVisitDoneWrapper(eval_env1())
  if args.first_visit_succ:
    config.first_visit_succ = True

  config.train_env = EnvModule(env, num_envs=args.num_envs, seed=args.seed)
  config.eval_env = EnvModule(eval_env, num_envs=args.num_eval_envs, name='eval_env', seed=args.seed + 1138)

  e = config.eval_env
  if args.alg.lower() == 'dqn':
    config.qvalue = PytorchModel('qvalue', lambda: Critic(FCBody(e.state_dim + e.goal_dim, args.layers, nn.LayerNorm, make_activ(config.activ)), e.action_dim))
  if args.alg.lower()  == 'clearning':
    config.actor = PytorchModel('actor',
                                lambda: Actor(FCBody(e.state_dim + e.goal_dim + 1, args.layers, nn.LayerNorm, make_activ(config.activ)), e.action_dim, e.max_action))
    config.critic = PytorchModel('critic',
                                lambda: Critic(FCBody(e.state_dim + e.goal_dim + 1 + e.action_dim, args.layers, nn.LayerNorm, make_activ(config.activ)), 1))
  else:
    config.actor = PytorchModel('actor',
                                lambda: Actor(FCBody(e.state_dim + e.goal_dim, args.layers, nn.LayerNorm, make_activ(config.activ)), e.action_dim, e.max_action))
    config.critic = PytorchModel('critic',
                                lambda: Critic(FCBody(e.state_dim + e.goal_dim + e.action_dim, args.layers, nn.LayerNorm, make_activ(config.activ)), 1))
    if args.alg.lower() == 'ddpg_explore':
      config.explore_actor = PytorchModel('explore_actor',
                                  lambda: Actor(FCBody(e.state_dim , args.layers, nn.LayerNorm, make_activ(config.activ)), e.action_dim, e.max_action))
      config.explore_critic = PytorchModel('explore_critic',
                                  lambda: Critic(FCBody(e.state_dim  + e.action_dim, args.layers, nn.LayerNorm, make_activ(config.activ)), 1))                            
    if args.alg.lower() == 'td3':
      config.critic2 = PytorchModel('critic2',
        lambda: Critic(FCBody(e.state_dim + e.goal_dim + e.action_dim, args.layers, nn.LayerNorm, make_activ(config.activ)), 1))

  if args.ag_curiosity == 'goaldisc' or args.ag_curiosity == 'minkde':
    config.goal_discriminator = PytorchModel('goal_discriminator', lambda: Critic(FCBody(e.state_dim + e.goal_dim, (256,256), nn.LayerNorm, make_activ(config.activ)), 1))

  if args.reward_module == 'env':
    config.goal_reward = GoalEnvReward()
  elif args.reward_module == 'intrinsic':
    config.goal_reward = NeighborReward()
    config.neighbor_embedding_network = PytorchModel('neighbor_embedding_network',
                                                     lambda: FCBody(e.goal_dim, (256, 256)))
  else:
    raise ValueError('Unsupported reward module: {}'.format(args.reward_module))

  if config.eval_env.goal_env:
    if not (args.first_visit_done or args.first_visit_succ):
      config.never_done = True  # NOTE: This is important in the standard Goal environments, which are never done

  agent = mrl.config_to_agent(config)
  agent.goaldim = args.goaldim
  agent.goalnum = args.goalnum
  
  agent.load('checkpoint'+str(args.load))
  agent.eval_mode()
  env = agent.env
  state = env.reset()
  agent.reset_idxs = []
  agent.env_steps = 0
  # generate data at a static point
  
  for i in range (args.testtrain):
    #agent.train(5000,dont_train=True, dont_optimize=True)
    """
    res = np.mean(agent.eval(num_episodes=100).rewards)
    print(res)
    for _ in range(5000):
        
        state = env.reset()
        done = [False, False]
        while not all(done):

          action = agent.policy(state)
          next_state, reward, done, info = env.step(action)
          state, experience = debug_vectorized_experience(state, action, next_state, reward, done, info)
          agent.process_experience(experience)
    """
    res = np.mean(agent.eval(num_episodes=100).rewards)
    print(res)
    num_steps = 20000

    
    for step in range(num_steps // env.num_envs):
      
      agent.step = step * env.num_envs
      agent.total_step = num_steps
      #state['horizon'] = np.zeros(state['achieved_goal'].shape)+ self.config.other_args['env_max_step']
      
      action = agent.policy(state)
      next_state, reward, done, info = env.step(action)
      #state['horizon'] = state['horizon'] -1

      if agent.reset_idxs:
        env.reset(agent.reset_idxs)
        for i in agent.reset_idxs:
          done[i] = True
          if not 'done_observation' in info[i]:
            if isinstance(next_state, np.ndarray):
              info[i].done_observation = next_state[i]
            else:
              for key in next_state:
                info[i].done_observation = {k: next_state[k][i] for k in next_state}
        next_state = env.state
        agent.reset_idxs = []

      state, experience = debug_vectorized_experience(state, action, next_state, reward, done, info)
      agent.process_experience(experience)

  replay_buffer_v = OnlineHERBufferValidation()
  agent.set_module("replay_buffer_validation", replay_buffer_v)

  agent.module_dict["replay_buffer_validation"] = replay_buffer_v    
  for step in range(80000 // env.num_envs):
      
    agent.step = step * env.num_envs
    agent.total_step = num_steps
    #state['horizon'] = np.zeros(state['achieved_goal'].shape)+ self.config.other_args['env_max_step']
    
    action = agent.policy(state)
    next_state, reward, done, info = env.step(action)
    #state['horizon'] = state['horizon'] -1

    if agent.reset_idxs:
      env.reset(agent.reset_idxs)
      for i in agent.reset_idxs:
        done[i] = True
        if not 'done_observation' in info[i]:
          if isinstance(next_state, np.ndarray):
            info[i].done_observation = next_state[i]
          else:
            for key in next_state:
              info[i].done_observation = {k: next_state[k][i] for k in next_state}
      next_state = env.state
      agent.reset_idxs = []

    state, experience = debug_vectorized_experience(state, action, next_state, reward, done, info)
    agent.replay_buffer_validation.process_experience(experience)
    
  # train a new goal success predictor on it
  predictorlen = 800000
  staticsuccessPerdictor = GoalSuccessPredictor(batch_size=args.succ_bs, history_length= predictorlen, optimize_every=1)
  agent.set_module("staticSP", staticsuccessPerdictor)
  setattr(agent.staticSP, "logloss", 1)
  agent.module_dict["staticSP"] = staticsuccessPerdictor
  
  for i in range(5000):
    agent.staticSP._optimize()
    if i %10 ==0:
      agent.staticSP.validate()
  

  # test it 
  agent.eval_mode()
  env = agent.eval_env
  num_envs = env.num_envs
  
  episode_rewards, episode_steps = [], []
  discounted_episode_rewards = []
  is_successes = []
  record_success = False
  import csv
  f = open('plotstatic.csv', 'a+')
  writer = csv.writer(f)
  print('test')
  while len(episode_rewards) < 1000:
    state = env.reset()
    qstate = np.concatenate((state['observation'], state['desired_goal']), -1)
    obs = state['observation']
    dgoal = state['desired_goal']
    dones = np.zeros((num_envs,))
    steps = np.zeros((num_envs,))
    is_success = np.zeros((num_envs,))
    ep_rewards = [[] for _ in range(num_envs)]

    while not np.all(dones):
      action = agent.policy(state)
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
    #print('testdone')
    if hasattr(agent, 'state_normalizer'):
      qstate = agent.state_normalizer(qstate)
      
    q_values = agent.ag_curiosity.compute_q(qstate)

    ags = state['desired_goal']
    #print(ags)
    nobuf= False
    density_module = agent.ag_kde
    if not density_module.ready:
      buffer = getattr(density_module, density_module.buffer_name).buffer.BUFF['buffer_' + density_module.item]
      if len(buffer)==0:
        nobuf= True
      else:
        density_module._optimize(force=True)
    num_envs = 1
    num_sampled_ags = ags.shape[0]
    predictions = agent.staticSP(obs,dgoal).reshape(num_envs, num_sampled_ags) 
    # score the sampled_ags to get log densities, and exponentiate to get densities
    flattened_sampled_ags = ags.reshape(num_envs * num_sampled_ags, -1)
    if not nobuf:
      sampled_ag_scores = density_module.evaluate_log_density(flattened_sampled_ags)
      
    else:
      sampled_ag_scores= [-1]
    #row = [str(i) for i in is_success] + ['score'] + [str(i) for i in q_values]
    #writer.writerow(row)
      
    j=0
    for ep_reward, step, is_succ in zip(ep_rewards, steps, is_success):
      if record_success:
        is_successes.append(is_succ)
      episode_rewards.append(sum(ep_reward))
      discounted_episode_rewards.append(discounted_sum(ep_reward, agent.config.gamma))
      episode_steps.append(step)
      #row = [str(discounted_sum(ep_reward, self.config.gamma))]+ ['score'] + [str(predictions[i])]
      row = [str(is_succ)]+ ['score'] + [str(predictions[j])]
      j +=1
      writer.writerow(row)

    
    
  f.close()
  if hasattr(agent, 'logger'):
    if len(is_successes):
      agent.logger.add_scalar('Test/Success', np.mean(is_successes))
    agent.logger.add_scalar('Test/Episode_rewards', np.mean(episode_rewards))
    agent.logger.add_scalar('Test/Discounted_episode_rewards', np.mean(discounted_episode_rewards))
    agent.logger.add_scalar('Test/Episode_steps', np.mean(episode_steps))
  return AttrDict({
    'rewards': episode_rewards,
    'steps': episode_steps
  })



if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(description="Train DDPG", formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=100, width=120))
  parser.add_argument('--parent_folder', default='/tmp/test_mega', type=str, help='where to save progress')
  parser.add_argument('--prefix', type=str, default='proto', help='Prefix for agent name (subfolder where it is saved)')
  parser.add_argument('--env', default="FetchPush-v1", type=str, help="gym environment")
  parser.add_argument('--max_steps', default=5000000, type=int, help="maximum number of training steps")
  parser.add_argument('--alg', default='DDPG', type=str, help='algorithm to use (DDPG or TD3)')
  parser.add_argument(
      '--layers', nargs='+', default=(512,512,512), type=int, help='sizes of layers for actor/critic networks')
  parser.add_argument('--noise_type', default='Gaussian', type=str, help='type of action noise (Gaussian or OU)')
  parser.add_argument('--tb', default='', type=str, help='a tag for the agent name / tensorboard')
  parser.add_argument('--epoch_len', default=5000, type=int, help='number of steps between evals')
  parser.add_argument('--num_envs', default=None, type=int, help='number of envs')

  # Make env args
  parser.add_argument('--eval_env', default='', type=str, help='evaluation environment')
  parser.add_argument('--test_with_internal', default=True, type=bool, help='test with internal reward fn')
  parser.add_argument('--reward_mode', default=0, type=int, help='reward mode')
  parser.add_argument('--env_max_step', default=50, type=int, help='max_steps_env_environment')
  parser.add_argument('--per_dim_threshold', default='0.', type=str, help='per_dim_threshold')
  parser.add_argument('--hard', action='store_true', help='hard mode: all goals are high up in the air')
  parser.add_argument('--pp_in_air_percentage', default=0.5, type=float, help='sets in air percentage for fetch pick place')
  parser.add_argument('--pp_min_air', default=0.2, type=float, help='sets the minimum height in the air for fetch pick place when in hard mode')
  parser.add_argument('--pp_max_air', default=0.45, type=float, help='sets the maximum height in the air for fetch pick place')
  parser.add_argument('--train_dt', default=0., type=float, help='training distance threshold')
  parser.add_argument('--slow_factor', default=1., type=float, help='slow factor for moat environment; lower is slower. ')

  # Other args
  parser.add_argument('--first_visit_succ', action='store_true', help='Episodes are successful on first visit (soft termination).')
  parser.add_argument('--first_visit_done', action='store_true', help='Episode terminates upon goal achievement (hard termination).')
  parser.add_argument('--ag_curiosity', default=None, help='the AG Curiosity model to use: {minq, randq, minkde}')
  parser.add_argument('--bandwidth', default=0.1, type=float, help='bandwidth for KDE curiosity')
  parser.add_argument('--kde_kernel', default='gaussian', type=str, help='kernel for KDE curiosity')
  parser.add_argument('--num_sampled_ags', default=100, type=int, help='number of ag candidates sampled for curiosity')
  parser.add_argument('--alpha', default=-1.0, type=float, help='Skewing parameter on the empirical achieved goal distribution. Default: -1.0')
  parser.add_argument('--reward_module', default='env', type=str, help='Reward to use (env or intrinsic)')
  parser.add_argument('--save_embeddings', action='store_true', help='save ag embeddings during training?')
  parser.add_argument('--succ_bs', default=100, type=int, help='success predictor batch size')
  parser.add_argument('--succ_hl', default=200, type=int, help='success predictor history length')
  parser.add_argument('--succ_oe', default=250, type=int, help='success predictor optimize every')
  parser.add_argument('--ag_pred_ehl', default=5, type=int, help='achieved goal predictor number of timesteps from end to consider in episode')
  parser.add_argument('--transition_to_dg', action='store_true', help='transition to the dg distribution?')
  parser.add_argument('--no_cutoff', action='store_true', help="don't use the q cutoff for curiosity")
  parser.add_argument('--visualize_trained_agent', action='store_true', help="visualize the trained agent")
  parser.add_argument('--intrinsic_visualization', action='store_true', help="if visualized agent should act intrinsically; requires saved replay buffer!")
  parser.add_argument('--keep_dg_percent', default=-1e-1, type=float, help='Percentage of time to keep desired goals')
  parser.add_argument('--prioritized_mode', default='none', type=str, help='Modes for prioritized replay: none, mep (default: none)')
  parser.add_argument('--no_ag_kde', action='store_true', help="don't track ag kde")

  parser.add_argument('--ikde', action='store_true', help="Use independent KDE density instead of joint KDE")
  parser.add_argument('--goaldim', default=2, type=int, help='dimensions of goals in the environment')
  parser.add_argument('--goalnum', default=3, type=int, help='number of goals in the environment')
  parser.add_argument('--ga', action='store_true', help="make gamma 0.99")
  parser.add_argument('--transport', action='store_true', help="Use transport")
  parser.add_argument('--load', default=100, type=int, help="the checkpoint to load")
  parser.add_argument('--testtrain', default=10000, type=int, help="the amount of addtional training for the success predictor")
  parser = add_config_args(parser, config)
  args = parser.parse_args()

  import subprocess, sys
  args.launch_command = sys.argv[0] + ' ' + subprocess.list2cmdline(sys.argv[1:])

  main(args)
