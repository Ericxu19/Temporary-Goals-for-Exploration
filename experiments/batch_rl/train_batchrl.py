"""
Train a batch-rl agent on offline data. Can either pass env as one of the envs from the `d4rl`, or as a standard env, 
along with a path to a folder containing saved replay buffer to load. 

E.g. 1, using an d4rl environment:

  PYTHONPATH=./ python experiments/batch_rl/train_batchrl.py --parent_folder /home/silviu/Documents/batchrl/ --env halfcheetah-random-v0
  PYTHONPATH=./ python experiments/batch_rl/train_batchrl.py --parent_folder /home/silviu/Documents/batchrl/ --env hopper-random-v0
  PYTHONPATH=./ python experiments/batch_rl/train_batchrl.py --parent_folder /home/silviu/Documents/batchrl/ --env walker2d-random-v0

E.g. 2, using a saved replay buffer:

  PYTHONPATH=./ python experiments/batch_rl/train_batchrl.py --parent_folder /home/silviu/Documents/batchrl/ --env HalfCheetah-v3 --buffer_load_path /home/silviu/Documents/batchrl/batchrl_env-HalfCheetah-v3_alg-sac_seed0_tb-collected/checkpoint/

The default config overfits and performance is quite random. On the random offline datasets:
  HalfCheetah gets to around 3750 at its peak (max > 4000)
  Walker gets to around 270 at its peak (max > 400)
  Hopper gets to around 60 at its peak (max ~220)


"""

from mrl.import_all import *
from mrl.configs.make_continuous_agents import *
from envs.d4rl.get_datasets import DATASET_NAMES
from experiments.mega.make_env import make_env
from private.modules.op_agent import *

import time
import os
import gym
import d4rl
from gym.wrappers import TimeLimit
import numpy as np
import torch

def load_replay_buffer(agent, load_path=None):
  if agent.config.other_args['env'] in DATASET_NAMES:
    dummy_env = gym.make(agent.config.other_args['env'])
    dataset = dummy_env.get_dataset()
    dummy_env.close()

    dataset = (
      dataset['observations'][:-1],
      dataset['actions'][:-1],
      dataset['rewards'][:-1].reshape(-1, 1),
      dataset['observations'][1:],
      dataset['terminals'][:-1].reshape(-1, 1),
    )

    agent.replay_buffer.buffer.add_batch(*dataset)
  else:
    assert load_path is not None
    agent.replay_buffer.load(load_path)

def main(config, args):

  # use the old replay buffer, since it adds experience to buffer immediately and don't need HER
  del config.module_replay
  config.module_replay = OldReplayBuffer()

  # dont use state normalizer
  del config.module_state_normalizer

  torch.set_num_threads(min(4, args.num_envs))
  torch.set_num_interop_threads(min(4, args.num_envs))

  # create agent
  agent  = mrl.config_to_agent(config)
  
  # load the offline data
  load_replay_buffer(agent, args.buffer_load_path)
  
  res = np.mean(agent.eval(num_episodes=10).rewards)
  agent.logger.log_color('Initial test reward (10 eps):', '{:.2f}'.format(res))

  for epoch in range(int(args.max_steps // args.epoch_len)):
    t = time.time()
    agent.train_mode()
    for _ in range(args.epoch_len):
      agent.config.env_steps +=1 # logger uses env_steps to decide when to write scalars.
      agent.config.opt_steps += 1
      agent.algorithm._optimize()
    agent.eval_mode()

    # EVALUATE
    res = np.mean(agent.eval(num_episodes=10).rewards)
    agent.logger.log_color('Test reward (10 eps):', '{:.2f}'.format(res))
    agent.logger.log_color('Epoch time:', '{:.2f}'.format(time.time() - t), color='yellow')

    # Don't save the agent... 
    # print("Saving agent at epoch {}".format(epoch))
    # agent.save('checkpoint')

# 3. Declare args for modules (also parent_folder is required!)
if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(description="Train DDPG", formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=100, width=120))
  parser.add_argument('--parent_folder', default='./batchrl_results', type=str, help='where to save progress')
  parser.add_argument('--prefix', type=str, default='train_batchrl', help='Prefix for agent name (subfolder where it is saved)')
  parser.add_argument('--env', default="HalfCheetah-v2", type=str, help="gym environment")
  parser.add_argument('--max_steps', default=1200000, type=int, help="maximum number of training steps")
  parser.add_argument('--alg', default='sac', type=str, help='algorithm to use (DDPG or TD3)')
  parser.add_argument('--layers', nargs='+', default=(512, 512, 512), type=int, help='hidden layers for actor/critic')
  parser.add_argument('--tb', default='_', type=str, help='a tag for the agent name / tensorboard')
  parser.add_argument('--epoch_len', default=500, type=int, help='number of steps between evals')
  parser.add_argument('--num_envs', default=1, type=int, help='number of envs')
  parser.add_argument('--num_eval_envs', default=5, type=int, help='number of eval envs')
  parser.add_argument('--buffer_load_path', default=None, type=str, help='path to folder holding desired replay_buffer.pickle')


  parser.add_argument('--op_coef', default=0.2, type=float, help='op loss coefficient')
  parser.add_argument('--cql_coef', default=0.2, type=float, help='op loss coefficient')
  parser.add_argument('--cql_tau', default=0., type=float, help='op loss coefficient')
  parser.add_argument('--bc_loss', default=False, type=str2bool, nargs='?', const=True)
  parser.add_argument('--detach_policy', default=True, type=str2bool, nargs='?', const=True)


  args, unknown = parser.parse_known_args()

  if args.alg.lower() == 'ddpg':
    make = make_ddpg_agent
    config = spinning_up_ddpg_config()
  elif args.alg.lower() == 'td3':
    make = make_td3_agent
    config = spinning_up_td3_config()
  elif args.alg.lower() == 'sac':
    make = make_sac_agent
    config = spinning_up_sac_config()
  elif args.alg.lower() == 'op':
    make = make_td3_agent
    config = spinning_up_td3_config()
  config.batch_size = 2000
  config.critic_lr = 1e-3
  config.actor_lr = 3e-4
  config.td3_noise = 0.1
  config.replay_size = int(2.5e6)
  config.warm_up = 0
  config.use_actor_target = True
  config.activ = 'gelu'
  config.op_loss = 'huber'
  config.cql_loss = 'intlog'
  config.detach_policy = True
  config.bound_constraint_coef = 1.

  config.op_coef = 0.2
  config.cql_coef = 0.2
  config.unif_repeats = 0
  config.poly_repeats = 1
  config.bc_loss = False
  config.a_noise = 0.5
  config.a_input_noise = 0.5
  config.cql_tau = 10.

  parser = add_config_args(parser, config)
  args = parser.parse_args()

  import subprocess, sys
  args.launch_command = sys.argv[0] + ' ' + subprocess.list2cmdline(sys.argv[1:])

  config = make(base_config=config, args=args, agent_name_attrs=['env', 'alg', 'td3_noise', 'op_coef', 'cql_coef', 'cql_tau', 'detach_policy', 'actor_lr', 'critic_lr', 'unif_repeats', 'poly_repeats', 'a_noise', 'a_input_noise', 'seed', 'tb'])

  if args.alg.lower() == 'op':
    del config.module_algorithm
    config.algorithm = OP_Hard()

  main(config, args)