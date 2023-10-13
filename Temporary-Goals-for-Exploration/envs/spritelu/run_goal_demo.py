# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# python2 python3
"""Start demo GUI for Spriteworld task configs.

To play a task, run this on the task config:
```bash
python run_demo.py --config=$path_to_task_config$
```

Be aware that this demo overrides the action space and renderer for ease of
playing, so those will be different from what are specified in the task config.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
from absl import app
from absl import flags
from spriteworld import demo_ui, demo_goal_ui
from spriteworld import renderers as spriteworld_renderers, factor_distributions as distribs
from spriteworld import environment, sprite, tasks, action_spaces, sprite_generators
FLAGS = flags.FLAGS
TERMINATE_DISTANCE = 0.05
def image_renderers():
  return {
    'observation': spriteworld_renderers.PILRenderer((64,64), anti_aliasing=5),
    'achieved_goal': spriteworld_renderers.AchievedPILGoalRenderer((64,64), anti_aliasing=5),
    'desired_goal': spriteworld_renderers.PILGoalRenderer((64, 64), anti_aliasing=5)
  }

flags.DEFINE_string('config', 'spriteworld.configs.protoge.goal_general',
                    'Module name of task config to use.')
flags.DEFINE_boolean('test_config', False,
                     'Whether to use test config (for prototyping configs).')
flags.DEFINE_integer('render_size', 256,
                     'Height and width of the output image.')
flags.DEFINE_integer('anti_aliasing', 10, 'Renderer anti-aliasing factor.')

flags.DEFINE_integer('num_goal_objects', 1, 'Number of goal objects.')
flags.DEFINE_integer('num_barriers', 1, 'Number of barrier objects.')
flags.DEFINE_integer('num_distractors', 0, 'Number of distractor objects.')
flags.DEFINE_boolean('agent_has_goal', False, 'Whether agent has a goal position')
flags.DEFINE_float('scale', 0.1, 'length-scale of each sprite')


""" FOR PROTOTYPING CONFIGS """

from spriteworld import environment, renderers, sprite, tasks, action_spaces
import os, copy

GOAL_ENV_IMAGE_RENDERERS = {
  'observation': renderers.PILRenderer((100,100)),
  'achieved_goal': renderers.AchievedPILGoalRenderer((100,100)),
  'desired_goal': renderers.PILGoalRenderer((100, 100))
}

s1 = sprite.Sprite(0.25, 0.25, 'triangle', c0=140,c1=220, c2=80, goal_x=0.1, goal_y=0.1)
s2 = sprite.Sprite(0.9, 0.5, 'square', c0=200,c2=255)
s3 = sprite.Sprite(0.25, 0.75, 'circle', c2=255, c1=210, goal_x=0.1, goal_y=0.9)
s4 = sprite.Sprite(0.50, 0.75, 'star_5', c0=255, c1=80)

b1 = sprite.Sprite(0.4, 0.4, is_barrier=True)
b2 = sprite.Sprite(0.3, 0.4, is_barrier=True)
b3 = sprite.Sprite(0.2, 0.4, is_barrier=True)

init_sprites = lambda: copy.deepcopy((b1, b2, b3, s1, s2, s3, s4))
simple_sprites = lambda: copy.deepcopy((s3, s4))

test_config = {
    'task': tasks.SparseGoalPlacement(),
    'action_space': action_spaces.Navigate(),
    'renderers': GOAL_ENV_IMAGE_RENDERERS,
    'init_sprites': init_sprites,
    'max_episode_length': 1000,
    'metadata': {
        'name': os.path.basename(__file__)
    }
}

def get_config(num_goal_objects=1, num_barriers=0, num_distractors=0, agent_has_goal=False, scale=0.1):
  """Generate environment config.

  Args:
    num_goal_objects: number of objects that need to be placed
    num_barriers: number of random barriers to spawn
    num_distractors: number of distractor objects (do not need to be placed)
    agent_has_goal: Whether or not the agent also needs to be placed at a target location

  Returns:
    config: Dictionary defining task/environment configuration. Can be fed as
      kwargs to environment.Environment.
  """
  gen_list = []

  shared_factors = distribs.Product([
    distribs.Continuous('x', 0.1, 0.9),
    distribs.Continuous('y', 0.1, 0.9),
    distribs.Continuous('c0', 25, 230),
    distribs.Continuous('c1', 25, 230),
    distribs.Continuous('c2', 25, 230),
    distribs.Discrete('scale', [scale])
  ])

  goal_loc = distribs.Product([
    distribs.Continuous('goal_x', 0.1, 0.9),
    distribs.Continuous('goal_y', 0.1, 0.9),
  ])

  ## OBJECTS
  goal_factors = distribs.Product([
    shared_factors,
    distribs.Discrete('shape', ['square', 'triangle', 'circle']),
    goal_loc
  ])
  goal_sprite_gen = sprite_generators.generate_static_sprites(goal_factors, num_sprites=num_goal_objects)
  gen_list.append(goal_sprite_gen )

  ## BARRIERS
  barrier_factors = distribs.Product([
    shared_factors,
    distribs.Continuous('barrier_stretch', 2., 10.),
    distribs.Continuous('angle', 0., 360),
    distribs.Discrete('is_barrier', [True])
  ])
  barrier_sprite_gen = sprite_generators.generate_c_maze_barriers(barrier_factors, num_sprites=num_barriers)
  gen_list.append(barrier_sprite_gen)
  
  ## DISTRACTORS
  distractor_factors = distribs.Product([
    shared_factors,
    distribs.Discrete('shape', ['square', 'triangle', 'circle']),
  ])
  distractor_sprite_gen = sprite_generators.generate_sprites(distractor_factors, num_sprites=num_distractors)
  gen_list.append(distractor_sprite_gen)

  ## AGENT
  if agent_has_goal:
    agent_factors = distribs.Product([
      shared_factors,
      distribs.Discrete('shape', ['star_5']),
      goal_loc,
    ])
  else:
    agent_factors = distribs.Product([
      shared_factors,
      distribs.Discrete('shape', ['star_5']),
    ])

  #agent_sprite_gen = sprite_generators.generate_sprites(agent_factors, num_sprites=1)
  agent_sprite_gen = sprite_generators.generate_c_agent_sprites(agent_factors)

  sprite_gen = sprite_generators.chain_generators(*gen_list)

  # Randomize sprite ordering to eliminate any task information from occlusions
  # sprite_gen = sprite_generators.shuffle(sprite_gen)
  
  # Add the agent in at the end
  sprite_gen = sprite_generators.resample_if_in_barrier(
    sprite_generators.chain_generators(sprite_gen, agent_sprite_gen))

  config = {
    'task': tasks.SparseGoalPlacement(epsilon=TERMINATE_DISTANCE),
    'action_space': action_spaces.Navigate(step_size=0.04),
    'renderers': image_renderers(),
    'init_sprites': sprite_gen,
    'max_episode_length': 150,
    'metadata': {
        'name': os.path.basename(__file__),
    }
  }

  return config
def main(_):
  if FLAGS.test_config:
    demo_goal_ui.setup_run_ui(test_config, FLAGS.render_size, False, FLAGS.anti_aliasing)
  else:
    config = get_config(num_goal_objects=0, num_barriers=0, num_distractors=0, agent_has_goal=True, scale=0.05)
    demo_goal_ui.setup_run_ui(config, FLAGS.render_size, False, FLAGS.anti_aliasing)

if __name__ == '__main__':
  app.run(main)
