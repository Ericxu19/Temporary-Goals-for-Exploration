"""
This is just a basic online RL training script / model for a training script. 
"""

from mrl.import_all import *
from mrl.configs.make_continuous_agents import *
from experiments.mega.make_env import make_env

import time
import os
import gym
import numpy as np
import torch

config = spinning_up_sac_config()

def main(args):
  if args.alg.lower() == 'ddpg':
    make = make_ddpg_agent
    conf = spinning_up_ddpg_config
  elif args.alg.lower() == 'td3':
    make = make_td3_agent
    conf = spinning_up_td3_config
  elif args.alg.lower() == 'sac':
    make = make_sac_agent
    conf = spinning_up_sac_config

  config = make(base_config=conf, args=args, agent_name_attrs=['env', 'alg', 'seed', 'tb'])

  # use the old replay buffer, since it adds experience to buffer immediately and don't need HER
  del config.module_replay
  config.module_replay = OldReplayBuffer()

  # dont use state normalizer on basic mujoco task
  del config.module_state_normalizer

  torch.set_num_threads(min(4, args.num_envs))
  torch.set_num_interop_threads(min(4, args.num_envs))

  agent  = mrl.config_to_agent(config)
  
  res = np.mean(agent.eval(num_episodes=30).rewards)
  agent.logger.log_color('Initial test reward (10 eps):', '{:.2f}'.format(res))

  for epoch in range(int(args.max_steps // args.epoch_len)):
    t = time.time()
    agent.train(num_steps=args.epoch_len)

    # EVALUATE
    res = np.mean(agent.eval(num_episodes=30).rewards)
    agent.logger.log_color('Test reward (10 eps):', '{:.2f}'.format(res))
    agent.logger.log_color('Epoch time:', '{:.2f}'.format(time.time() - t), color='yellow')

    print("Saving agent at epoch {}".format(epoch))
    agent.save('checkpoint')

# 3. Declare args for modules (also parent_folder is required!)
if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(description="Train DDPG", formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=100, width=120))
  parser.add_argument('--parent_folder', default='/tmp/test_train_online_agent', type=str, help='where to save progress')
  parser.add_argument('--prefix', type=str, default='batchrl', help='Prefix for agent name (subfolder where it is saved)')
  parser.add_argument('--env', default="HalfCheetah-v2", type=str, help="gym environment")
  parser.add_argument('--max_steps', default=1000000, type=int, help="maximum number of training steps")
  parser.add_argument('--alg', default='td3', type=str, help='algorithm to use (DDPG or TD3)')
  parser.add_argument('--layers', nargs='+', default=(256, 256), type=int, help='hidden layers for actor/critic')
  parser.add_argument('--tb', default='', type=str, help='a tag for the agent name / tensorboard')
  parser.add_argument('--epoch_len', default=5000, type=int, help='number of steps between evals')
  parser.add_argument('--num_envs', default=None, type=int, help='number of envs (defaults to procs - 1)')

  parser = add_config_args(parser, config)
  args = parser.parse_args()

  import subprocess, sys
  args.launch_command = sys.argv[0] + ' ' + subprocess.list2cmdline(sys.argv[1:])

  main(args)
