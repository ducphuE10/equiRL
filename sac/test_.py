# from sac.train import run_task
from softgym.registered_env import env_arg_dict
import numpy as np
import torch
import os
import time
import json
import copy

from sac import utils
from sac.logger import Logger

from softagent_copy.sac.sac_agent import CurlSacAgent
from sac.default_config import DEFAULT_CONFIG

from chester import logger
from envs.env import Env

from softgym.utils.visualization import save_numpy_as_gif, make_grid
import matplotlib.pyplot as plt

from train import vv_to_args, update_env_kwargs

reward_scales = {
    'PassWater': 20.0,
    'PourWater': 20.0,
    'ClothFold': 50.0,
    'ClothFlatten': 50.0,
    'ClothDrop': 50.0,
    'RopeFlatten': 50.0,
}

clip_obs = {
    'PassWater': None,
    'PourWater': None,
    'ClothFold': (-3, 3),
    'ClothFlatten': (-2, 2),
    'ClothDrop': None,
    'RopeFlatten': None,
}


def get_lr_decay(env_name, obs_mode):
    if env_name == 'RopeFlatten' or (env_name == 'ClothFlatten' and obs_mode == 'cam_rgb'):
        return 0.01
    elif obs_mode == 'point_cloud':
        return 0.01
    else:
        return None


def get_actor_critic_lr(env_name, obs_mode):
    if env_name == 'ClothFold' or (env_name == 'RopeFlatten' and obs_mode == 'point_cloud'):
        if obs_mode == 'cam_rgb':
            return 1e-4
        else:
            return 5e-4
    if obs_mode == 'cam_rgb':
        return 3e-4
    else:
        return 1e-3


def get_alpha_lr(env_name, obs_mode):
    if env_name == 'ClothFold':
        return 2e-5
    else:
        return 1e-3


def main():
    import argparse
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument('--exp_name', default='SAC', type=str)
    parser.add_argument('--env_name', default='ClothFlatten')
    parser.add_argument('--log_dir', default='./data/curl/')
    parser.add_argument('--test_episodes', default=10, type=int)
    parser.add_argument('--seed', default=100, type=int)
    parser.add_argument('--save_tb', default=False)  # Save stats to tensorbard
    parser.add_argument('--save_video', default=True)
    parser.add_argument('--save_model', default=True)  # Save trained models

    # CURL
    parser.add_argument('--alpha_fixed', default=False, type=bool)  # Automatic tuning of alpha
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--replay_buffer_capacity', default=100000)
    parser.add_argument('--batch_size', default=128)

    # Override environment arguments
    parser.add_argument('--env_kwargs_render', default=True, type=bool)  # Turn off rendering can speed up training
    parser.add_argument('--env_kwargs_camera_name', default='default_camera', type=str)
    parser.add_argument('--env_kwargs_observation_mode', default='cam_rgb', type=str)  # Should be in ['key_point', 'cam_rgb', 'point_cloud']
    parser.add_argument('--env_kwargs_num_variations', default=1, type=int)

    args = parser.parse_args()

    args.algorithm = 'CURL'

    # Set env_specific parameters

    env_name = args.env_name
    obs_mode = args.env_kwargs_observation_mode
    args.actor_lr = args.critic_lr = get_actor_critic_lr(env_name, obs_mode)
    args.lr_decay = get_lr_decay(env_name, obs_mode)
    args.scale_reward = reward_scales[env_name]
    args.clip_obs = clip_obs[env_name] if obs_mode == 'key_point' else None
    args.env_kwargs = env_arg_dict[env_name]



    vv = args.__dict__
    log_dir=None
    exp_name=None
    if log_dir or logger.get_dir() is None:
        logger.configure(dir=log_dir, exp_name=exp_name, format_strs=['csv'])
    logdir = logger.get_dir()
    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)
    
    updated_vv = copy.copy(DEFAULT_CONFIG)
    updated_vv.update(**vv)
    args = vv_to_args(updated_vv)


    args.__dict__ = update_env_kwargs(args.__dict__)


    symbolic = args.env_kwargs['observation_mode'] != 'cam_rgb'
    args.encoder_type = 'identity' if symbolic else 'pixel'

    print(symbolic) #TRUE
    print(args.encoder_type) #identity
    # env = Env(args.env_name, symbolic, args.seed, 200, 1, 8, args.pre_transform_image_size, env_kwargs=args.env_kwargs, normalize_observation=False,
    #           scale_reward=args.scale_reward, clip_obs=args.clip_obs)

    """ test xiu xiu """
    env_name = 'ClothFlatten'
    symbolic = False
    obs_mode = 'pixel'
    pre_transform_image_size = 128
    env_kwargs = env_arg_dict[env_name]
    scale_reward = reward_scales[env_name]
    clip_obs = clip_obs[env_name] if obs_mode == 'key_point' else None

    env = Env(env_name, symbolic, 100, 200, 1, 8, pre_transform_image_size, env_kwargs=env_kwargs, normalize_observation=False,
                scale_reward=scale_reward, clip_obs=clip_obs)

    action_shape = env.action_space.shape 
    obs_shape = env.observation_space.shape 

    print(obs_shape)
# if __name__ == '__main__':
#     # main()
#     from softagent_copy.sac.sacagent import Actor
#     # obs_shape, action_shape, hidden_dim, encoder_type,
#     #   encoder_feature_dim, log_std_min, log_std_max, num_layers, num_filters
#     x = Actor([1,1,100], 1, 1, 'pixel',100, 1, 1, 1, 32)
