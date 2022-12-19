import numpy as np
import torch
import os
import time
import json
import copy

from equi import utils
from equi.logger import Logger

from equi.equi_agent import SacAgent
from equi.default_config import DEFAULT_CONFIG

from chester import logger
from envs.env import Env

from softgym.utils.visualization import save_numpy_as_gif, make_grid
import matplotlib.pyplot as plt

from scipy.spatial import ConvexHull
import wandb
import gc

def update_env_kwargs(vv):
    new_vv = vv.copy()
    for v in vv:
        if v.startswith('env_kwargs_'):
            arg_name = v[len('env_kwargs_'):]
            new_vv['env_kwargs'][arg_name] = vv[v]
            del new_vv[v]
    return new_vv


def vv_to_args(vv):
    class VArgs(object):
        def __init__(self, vv):
            for key, val in vv.items():
                setattr(self, key, val)

    args = VArgs(vv)

    return args


def run_task(vv, log_dir=None, exp_name=None):
    updated_vv = copy.copy(DEFAULT_CONFIG)
    updated_vv.update(**vv)
    args = vv_to_args(updated_vv)
    if args.wandb:
        log_dir = os.path.join(log_dir, f's{args.wandb_seed}')
    if log_dir or logger.get_dir() is None:
        logger.configure(dir=log_dir, exp_name=exp_name, format_strs=['csv'])
    logdir = logger.get_dir()
    os.makedirs(logdir, exist_ok=True)
    assert logdir is not None
    with open(os.path.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(updated_vv, f, indent=2, sort_keys=True)
    main(args)

def get_info_stats(infos):
    # infos is a list with N_traj x T entries
    N = len(infos)
    T = len(infos[0])
    stat_dict_all = {key: np.empty([N, T], dtype=np.float32) for key in infos[0][0].keys()}
    for i, info_ep in enumerate(infos):
        for j, info in enumerate(info_ep):
            for key, val in info.items():
                stat_dict_all[key][i, j] = val

    stat_dict = {}
    for key in infos[0][0].keys():
        stat_dict[key + '_mean'] = np.mean(np.array(stat_dict_all[key]))
        stat_dict[key + '_final'] = np.mean(stat_dict_all[key][:, -1])
    return stat_dict


def evaluate(env, agent, video_dir, num_episodes, L, step, args):
    all_ep_rewards = []

    def run_eval_loop(sample_stochastically=True):
        start_time = time.time()
        prefix = 'stochastic_' if sample_stochastically else ''
        infos = []
        all_frames = []
        plt.figure()
        for i in range(num_episodes):
            obs = env.reset(eval_flag=True)
            done = False
            episode_reward = 0
            ep_info = []
            frames = [env.get_image(128, 128)]
            rewards = []
            while not done:
                if args.encoder_type == 'pixel':
                    if obs.shape[0] == 1:
                        obs = obs[0]
                    # print(obs.shape)
                    
                # with utils.eval_mode(agent):
                #     if sample_stochastically:
                #         action = agent.sample_action(obs)
                #     else:
                #         action = agent.select_action(obs)
                action = np.array([1.0, 0.0, 0.0, 0.0, ])
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                ep_info.append(info)
                frames.append(env.get_image(128, 128))
                rewards.append(reward)
            plt.plot(range(len(rewards)), rewards)
            if len(all_frames) < 8:
                all_frames.append(frames)
            infos.append(ep_info)

            L.log('eval/' + prefix + 'episode_reward', episode_reward, step)
            all_ep_rewards.append(episode_reward)
        plt.xlabel('Timestep')
        plt.ylabel('Reward')
        plt.title('Reward over time')
        plt.savefig(os.path.join(video_dir, '%d.png' % step))
        all_frames = np.array(all_frames).swapaxes(0, 1)
        all_frames = np.array([make_grid(np.array(frame), nrow=2, padding=3) for frame in all_frames])
        save_numpy_as_gif(all_frames, os.path.join(video_dir, '%d.gif' % step))

        for key, val in get_info_stats(infos).items():
            L.log('eval/info_' + prefix + key, val, step)
            if args.wandb:
                wandb.log({key:val},step = step)
        L.log('eval/' + prefix + 'eval_time', time.time() - start_time, step)
        mean_ep_reward = np.mean(all_ep_rewards)
        best_ep_reward = np.max(all_ep_rewards)
        L.log('eval/' + prefix + 'mean_episode_reward', mean_ep_reward, step)
        L.log('eval/' + prefix + 'best_episode_reward', best_ep_reward, step)

    run_eval_loop(sample_stochastically=False)
    L.dump(step)


def make_agent(obs_shape, action_shape, args, device):
    if args.agent == 'sac':
        
        return SacAgent(
            args=args,
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            alpha_fixed=args.alpha_fixed,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            log_interval=args.log_interval,
            detach_encoder=args.detach_encoder,
            curl_latent_dim=args.curl_latent_dim,
            num_rotations=args.num_rotations
        )
    else:
        assert 'agent is not supported: %s' % args.agent


def main(args):
    # import ipdb; ipdb.set_trace()
    torch.cuda.empty_cache()
    if args.seed == -1:
        args.__dict__["seed"] = np.random.randint(1, 1000000)
    utils.set_seed_everywhere(args.seed)

    args.__dict__ = update_env_kwargs(args.__dict__)  # Update env_kwargs
    
    if args.wandb:
        # ed44c646a708f75a7fe4e39aee3844f8bfe44858
        wandb.login(key='ed44c646a708f75a7fe4e39aee3844f8bfe44858')
        group_name = args.exp_name + '_aug' if args.aug_transition else args.exp_name + '_no_aug'
        wandb.init(project=args.env_name, settings=wandb.Settings(_disable_stats=True), group=group_name, name=f's{args.wandb_seed}', entity='longdinh')
    else:
        print('==================== NOT USING WANDB ====================')

    symbolic = False if args.env_kwargs['observation_mode'] in ['cam_rgb', 'img_depth', 'only_depth'] else True
    args.encoder_type = 'identity' if symbolic else 'pixel'

    env = Env(args.env_name, symbolic, args.seed, 100, 1, 8, args.pre_transform_image_size, env_kwargs=args.env_kwargs, normalize_observation=False,
              scale_reward=args.scale_reward, clip_obs=args.clip_obs)
    env.seed(args.seed)

    # make directory
    ts = time.gmtime()
    ts = time.strftime("%m-%d", ts)

    args.work_dir = logger.get_dir()
    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    buffer_dir = utils.make_dir(os.path.join(args.work_dir, 'buffer'))
    L = Logger(args.work_dir, use_tb=args.save_tb, chester_logger=logger)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    action_shape = env.action_space.shape
    if args.encoder_type == 'pixel':
        if args.env_kwargs['observation_mode'] == 'cam_rgb':
            obs_shape = (3, args.image_size, args.image_size)
        elif args.env_kwargs['observation_mode'] == 'only_depth':
            obs_shape = (1, args.image_size, args.image_size)
        else:
            obs_shape = (4, args.image_size, args.image_size)
        pre_aug_obs_shape = obs_shape
    else:
        obs_shape = env.observation_space.shape
        pre_aug_obs_shape = obs_shape

    if args.aug_transition:
        print(f'==================== AUGMENTED TRANSITION with {args.aug_n} TRANSFORMATION of {args.aug_type}====================')
        replay_buffer = utils.ReplayBufferAugmented(
            obs_shape=pre_aug_obs_shape,
            action_shape=action_shape,
            capacity=args.replay_buffer_capacity,
            batch_size=args.batch_size,
            device=device,
            image_size=args.image_size,
            aug_n = args.aug_n,
        )
    else:
        print('================ DON NOT USE AUGMENTED TRANSITION =================')
        replay_buffer = utils.ReplayBuffer(
            obs_shape=pre_aug_obs_shape,
            action_shape=action_shape,
            capacity=args.replay_buffer_capacity,
            batch_size=args.batch_size,
            device=device,
            image_size=args.image_size,
        )

    # agent = make_agent(
    #     obs_shape=obs_shape,
    #     action_shape=action_shape,
    #     args=args,
    #     device=device
    # )
    
    # episode, episode_reward, done, ep_info = 0, 0, True, []
    # start_time = time.time()
    # total_time = 0

    print('==================== START COLLECTING DEMONSTRATIONS ====================')
    all_frames_planner = []
    thresh = env.cloth_particle_radius + env.action_tool.picker_radius + env.action_tool.picker_threshold
    while True:
        obs = env.reset()
        episode_step = 0
        frames = [env.get_image(128, 128)]
        picker_pos, particle_pos = env.action_tool._get_pos()
        hull = ConvexHull(particle_pos[:, [0, 2]])
        bound_id = set()
        for simplex in hull.simplices:
            bound_id.add(simplex[0])
            bound_id.add(simplex[1])
        bound_id = list(bound_id)
        # choose 2 random boundary id with min distance > 10 * particle_radius
        flag_choose_id = True
        count_choose_id = 0
        while True:
            choosen_id = np.random.choice(bound_id, 2, replace=False)
            if np.linalg.norm(particle_pos[choosen_id[0], [0, 2]] - particle_pos[choosen_id[1], [0, 2]]) > 6 * env.cloth_particle_radius:
                break
            if count_choose_id > 10:
                flag_choose_id = False
                break
        if not flag_choose_id:
            continue

        # find the closest points for picker
        print('choosen_id', choosen_id)
        if np.linalg.norm(particle_pos[choosen_id[0], :3] - picker_pos[0]) > np.linalg.norm(particle_pos[choosen_id[1], :3] - picker_pos[0]):
            choosen_id = np.array([choosen_id[1], choosen_id[0]])
        print('after choosen_id', choosen_id)
        # move to two choosen boundary points and pick them
        flag_pick_boundary = True
        count_pick_boundary = 0    
        while True:
            picker_pos, particle_pos = env.action_tool._get_pos()
            target_pos = particle_pos[choosen_id, :3]
            dis = target_pos - picker_pos
            norm = np.linalg.norm(dis, axis=1)
            action = np.clip(dis, -0.08, 0.08) / 0.08
            # import ipdb; ipdb.set_trace()
            if norm[0] <= thresh and norm[1] <= thresh:
                action = np.concatenate([action, np.ones((2, 1))], axis=1).reshape(-1)
            else:
                action = np.concatenate([action, np.zeros((2, 1))], axis=1).reshape(-1)
            next_obs, reward, done, info = env.step(action)
            done_bool = 1 if episode_step + 1 == env.horizon else float(done)
            # replay_buffer.add(obs, action, reward, next_obs, done_bool)
            frames.append(env.get_image(128, 128))
            episode_step += 1
            obs = next_obs
            count_pick_boundary += 1
            if count_pick_boundary >= 6:
                flag_pick_boundary = False
                break
            if env.action_tool.picked_particles[0] is not None and env.action_tool.picked_particles[1] is not None:
                if len(set(particle_pos[env.action_tool.picked_particles, 3])) == 1:
                    print(f'collected in {episode_step} steps')
                    break
        if not flag_pick_boundary:
            continue
        # choose fling primitive or pick&drag primitive
        if np.random.rand() < 1.5:
            print('==================== PICK AND DRAG ====================')
            # fling primitive
            # first, move to the height 0.3
            target_pos = env.action_tool._get_pos()[0]
            target_pos[:, 1] = 0.3
            flag_move_height = True
            count_move_height = 0
            while True:
                picker_pos = env.action_tool._get_pos()[0]
                dis = target_pos - picker_pos
                norm = np.linalg.norm(dis, axis=1)
                action = np.clip(dis, -0.08, 0.08) / 0.08
                action = np.concatenate([action, np.ones((2, 1))], axis=1).reshape(-1)
                next_obs, reward, done, info = env.step(action)
                done_bool = 1 if episode_step + 1 == env.horizon else float(done)
                # replay_buffer.add(obs, action, reward, next_obs, done_bool)
                frames.append(env.get_image(128, 128))
                episode_step += 1
                obs = next_obs
                count_move_height += 1
                if count_move_height >= 6:
                    flag_move_height = False
                    break
                if norm[0] <= thresh and norm[1] <= thresh:
                    break
            if not flag_move_height:
                continue
            # second, stretch the cloth
            curr_pos = env.action_tool._get_pos()[0]
            init_pos = env._get_flat_pos()
            init_dis = np.linalg.norm(init_pos[choosen_id[0], [0, 2]] - init_pos[choosen_id[1], [0, 2]])
            curr_dis = np.linalg.norm(curr_pos[choosen_id[0], [0, 2]] - curr_pos[choosen_id[1], [0, 2]])
            denta = (curr_dis - init_dis) / 2
            if curr_pos[0, 0] > curr_pos[1, 0]:
                left = 1
                right = 0
            else:
                left = 0
                right = 1
            cos_phi = (curr_pos[right, 0] - curr_pos[left, 0]) / curr_dis
            sin_phi = (curr_pos[right, 2] - curr_pos[left, 2]) / curr_dis
            target_pos = copy.deepcopy(curr_pos)
            target_pos[left, 0] = curr_pos[left, 0] - denta * cos_phi
            target_pos[left, 2] = curr_pos[left, 2] - denta * sin_phi
            target_pos[right, 0] = curr_pos[right, 0] + denta * cos_phi
            target_pos[right, 2] = curr_pos[right, 2] + denta * sin_phi
            flag_stretch = True
            count_stretch = 0
            while True:
                picker_pos = env.action_tool._get_pos()[0]
                dis = target_pos - picker_pos
                norm = np.linalg.norm(dis, axis=1)
                action = np.clip(dis, -0.08, 0.08) / 0.08
                action = np.concatenate([action, np.ones((2, 1))], axis=1).reshape(-1)
                next_obs, reward, done, info = env.step(action)
                done_bool = 1 if episode_step + 1 == env.horizon else float(done)
                # replay_buffer.add(obs, action, reward, next_obs, done_bool)
                frames.append(env.get_image(128, 128))
                episode_step += 1
                obs = next_obs
                count_stretch += 1
                if count_stretch >= 6:
                    flag_stretch = False
                    break
                if norm[0] <= thresh and norm[1] <= thresh:
                    break
            if not flag_stretch:
                continue
            # third, fling the cloth
            # first, move to the height 0.3
            #         
        if len(frames) != 100:
            for i in range(100 - len(frames)-1):
                frames.append(env.get_image(128, 128))
        print(f'Number of frames: {len(frames)}')
        all_frames_planner.append(frames)
        if len(all_frames_planner) == 10:
            break
    all_frames_planner = np.array(all_frames_planner).swapaxes(0, 1)
    all_frames_planner = np.array([make_grid(np.array(frame), nrow=2, padding=3) for frame in all_frames_planner])
    save_numpy_as_gif(all_frames_planner, 'expert.gif')

    
    # for step in range(args.num_train_steps):
    #     # evaluate agent periodically
    #     if step % args.eval_freq == 0:
    #         L.log('eval/episode', episode, step)
    #         evaluate(env, agent, video_dir, args.num_eval_episodes, L, step, args)
    #         if args.save_model and (step % (args.eval_freq * 5) == 0):
    #             agent.save(model_dir, step)
    #         if args.save_buffer:
    #             replay_buffer.save(buffer_dir)
            
    #     if done:
    #         if step > 0:
    #             if step % args.log_interval == 0:
    #                 finish_time = time.time()
    #                 L.log('train/duration', finish_time - start_time, step)
    #                 if args.wandb:
    #                     total_time += (finish_time - start_time)
    #                     wandb.log({'Duration': total_time / 3600.}, step=step)
    #                 for key, val in get_info_stats([ep_info]).items():
    #                     L.log('train/info_' + key, val, step)
    #                 L.dump(step)
    #             start_time = time.time()
    #         if step % args.log_interval == 0:
    #             L.log('train/episode_reward', episode_reward, step)

    #         obs = env.reset()
    #         done = False
    #         ep_info = []
    #         episode_reward = 0
    #         episode_step = 0
    #         episode += 1
    #         if step % args.log_interval == 0:
    #             L.log('train/episode', episode, step)

        # sample action for data collection
        # if step < args.init_steps:
        #     action = env.action_space.sample()
        # else:
        #     with utils.eval_mode(agent):
        #         action = agent.sample_action(obs)
        # with utils.eval_mode(agent):
        #     action = agent.sample_action(obs)

        # run training update
        # if step >= args.init_steps:
            # s_u = time.time()
            # agent.update(replay_buffer, L, step)
            # print(f'update time: {time.time() - s_u}')
        # s_e = time.time()
        # next_obs, reward, done, info = env.step(action)
        # print(f'env step time: {time.time() - s_e}')
        # allow infinit bootstrap
        # ep_info.append(info)
        # done_bool = 0 if episode_step + 1 == env.horizon else float(done)
        # episode_reward += reward
        # replay_buffer.add(obs, action, reward, next_obs, done_bool)

        # obs = next_obs
        # episode_step += 1
