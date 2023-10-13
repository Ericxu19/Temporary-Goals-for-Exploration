import gym
from goal_grid import GoalGridWorldEnv

# grid_file = '2_room_9x9.txt'
# grid_file = 'room_5x5_empty.txt'
grid_file = 'kontrived_room.txt'
random_init_loc = False
use_xy_obs = False
env = GoalGridWorldEnv(use_xy_obs, grid_size=5, max_step=25,grid_file=grid_file,random_init_loc=random_init_loc)
obs = env.reset()
env.render()
# Act randomly
num_eps = 0
while(num_eps < 10):
    obs, reward, done, _ = env.step(env.action_space.sample())
    # action = int(input("What is the action?\n"))
    # obs, reward, done, _ = env.step(action)
    env.render()
    if done:
        print("Episode reward: {}".format(reward))
        obs = env.reset()
        env.render()
        num_eps += 1

import ipdb; ipdb.set_trace()
# Using X-Y observation
use_xy_obs = True
env = GoalGridWorldEnv(use_xy_obs, grid_size=5, max_step=25,grid_file=grid_file,random_init_loc=random_init_loc)
obs = env.reset()
env.render()
# Act randomly
num_eps = 0
while(num_eps < 10):
    obs, reward, done, _ = env.step(env.action_space.sample())
    # action = int(input("What is the action?\n"))
    # obs, reward, done, _ = env.step(action)
    env.render()
    if done:
        print("Episode reward: {}".format(reward))
        obs = env.reset()
        env.render()
        num_eps += 1