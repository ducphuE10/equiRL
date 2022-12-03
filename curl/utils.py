import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import gym
import os
from collections import deque
import random
from torch.utils.data import Dataset, DataLoader
import time
from skimage.util.shape import view_as_windows
from collections import deque
from scipy.ndimage import affine_transform
from curl.default_config import DEFAULT_CONFIG
from matplotlib import pyplot as plt
from PIL import Image

class ConvergenceChecker(object):
    def __init__(self, threshold, history_len):
        self.threshold = threshold
        self.history_len = history_len
        self.queue = deque(maxlen=history_len)
        self.converged = None

    def clear(self):
        self.queue.clear()
        self.converged = False

    def append(self, value):
        self.queue.append(value)

    def converge(self):
        if self.converged:
            return True
        else:
            losses = np.array(list(self.queue))
            self.converged = (len(losses) >= self.history_len) and (np.mean(losses[self.history_len // 2:]) > np.mean(
                losses[:self.history_len // 2]) - self.threshold)
            return self.converged


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def module_hash(module):
    result = 0
    for tensor in module.state_dict().values():
        result += tensor.sum().item()
    return result


def make_dir(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path


def preprocess_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2 ** bits
    assert obs.dtype == torch.float32
    if bits < 8:
        obs = torch.floor(obs / 2 ** (8 - bits))
    obs = obs / bins
    obs = obs + torch.rand_like(obs) / bins
    obs = obs - 0.5
    return obs


class ReplayBuffer(Dataset):
    """Buffer to store environment transitions."""

    def __init__(self, obs_shape, action_shape, capacity, batch_size, device, image_size=84, transform=None):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.image_size = image_size
        self.transform = transform
        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, done):

        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample_proprio(self):

        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]

        obses = torch.as_tensor(obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(
            next_obses, device=self.device
        ).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        return obses, actions, rewards, next_obses, not_dones


    def sample_cpc(self):

        start = time.time()
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]
        pos = obses.copy()

        obses = random_crop(obses, self.image_size)
        next_obses = random_crop(next_obses, self.image_size)
        pos = random_crop(pos, self.image_size)

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(
            next_obses, device=self.device
        ).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        pos = torch.as_tensor(pos, device=self.device).float()
        cpc_kwargs = dict(obs_anchor=obses, obs_pos=pos,
                          time_anchor=None, time_pos=None)

        return obses, actions, rewards, next_obses, not_dones, cpc_kwargs

    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save:self.idx],
            self.next_obses[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx]
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.next_obses[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.not_dones[start:end] = payload[4]
            self.idx = end

    def __getitem__(self, idx):
        idx = np.random.randint(
            0, self.capacity if self.full else self.idx, size=1
        )
        idx = idx[0]
        obs = self.obses[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]
        next_obs = self.next_obses[idx]
        not_done = self.not_dones[idx]

        if self.transform:
            obs = self.transform(obs)
            next_obs = self.transform(next_obs)

        return obs, action, reward, next_obs, not_done

    def __len__(self):
        return self.capacity

class ReplayBufferAugmented(ReplayBuffer):
    def __init__(self, obs_shape, action_shape, capacity, batch_size, device, image_size=84, transform=None, aug_n=9):
        super().__init__(obs_shape, action_shape, capacity, batch_size, device, image_size, transform)
        self.aug_n = aug_n
    
    def add(self, obs, action, reward, next_obs, done):
        super().add(obs, action, reward, next_obs, done)
        os.makedirs('augmented', exist_ok=True)
        plt.figure(figsize=(15, 15))
        plt.subplot(self.aug_n+1, 2, 1)
        plt.imshow(obs[0].numpy().transpose(1, 2, 0))
        plt.subplot(self.aug_n+1, 2, 2)
        plt.imshow(next_obs[0].numpy().transpose(1, 2, 0))
        a = None
        for i in range(self.aug_n):
            obs_, action_, reward_, next_obs_, done_ = augmentTransition(obs, action, reward, next_obs, done, DEFAULT_CONFIG['aug_type'])
            plt.subplot(self.aug_n+1, 2, 2*i+3)
            plt.imshow(obs_[0].numpy().transpose(1, 2, 0))
            plt.subplot(self.aug_n+1, 2, 2*i+4)
            plt.imshow(next_obs_[0].numpy().transpose(1, 2, 0))
            super().add(obs_, action_, reward_, next_obs_, done_)
            a = action_
        plt.title(f'{action}---{a}')
        plt.savefig(f'augmented/{self.idx}.png')

def augmentTransition(obs, action, reward, next_obs, done, aug_type):
    if aug_type=='se2':
        return augmentTransitionSE2(obs, action, reward, next_obs, done)
    elif aug_type=='so2':
        return augmentTransitionSO2(obs, action, reward, next_obs, done)
    elif aug_type=='trans':
        return augmentTransitionTranslate(obs, action, reward, next_obs, done)
    elif aug_type=='shift':
        return augmentTransitionShift(obs, action, reward, next_obs, done)
    else:
        raise NotImplementedError

def augmentTransitionSE2(obs, action, reward, next_obs, done):
    dxy = action[::2].copy()
    dxy1, dxy2 = np.split(dxy, 2)
    obs, next_obs, dxy1, dxy2, transform_params = perturb(obs[0].numpy().copy(),
                                                          next_obs[0].numpy().copy(),
                                                          dxy1, dxy2)
    obs = obs.reshape(1, *obs.shape)
    obs = torch.from_numpy(obs)
    next_obs = next_obs.reshape(1, *next_obs.shape)
    next_obs = torch.from_numpy(next_obs)
    action = action.copy()
    action[0] = dxy1[0]
    action[2] = dxy1[1]
    action[4] = dxy2[0]
    action[6] = dxy2[1]
    return obs, action, reward, next_obs, done
    
def augmentTransitionSO2(obs, action, reward, next_obs, done):
    dxy = action[::2].copy()
    dxy1, dxy2 = np.split(dxy, 2)
    obs, next_obs, dxy1, dxy2, transform_params = perturb(obs[0].numpy().copy(),
                                                          next_obs[0].numpy().copy(),
                                                          dxy1, dxy2,
                                                          set_trans_zero=True)
    obs = obs.reshape(1, *obs.shape)
    obs = torch.from_numpy(obs)
    next_obs = next_obs.reshape(1, *next_obs.shape)
    next_obs = torch.from_numpy(next_obs)
    action = action.copy()
    action[0] = dxy1[0]
    action[2] = dxy1[1]
    action[4] = dxy2[0]
    action[6] = dxy2[1]
    return obs, action, reward, next_obs, done

def augmentTransitionTranslate(obs, action, reward, next_obs, done):
    dxy = action[::2].copy()
    dxy1, dxy2 = np.split(dxy, 2)
    obs, next_obs, dxy1, dxy2, transform_params = perturb(obs[0].numpy().copy(),
                                                          next_obs[0].numpy().copy(),
                                                          dxy1, dxy2,
                                                          set_theta_zero=True)
    obs = obs.reshape(1, *obs.shape)
    obs = torch.from_numpy(obs)
    next_obs = next_obs.reshape(1, *next_obs.shape)
    next_obs = torch.from_numpy(next_obs)
    return obs, action, reward, next_obs, done

def augmentTransitionShift(obs, action, reward, next_obs, done):
    heightmap_size = obs.shape[-1]
    padded_obs = transforms.Pad((4, 4, 4, 4), padding_mode='edge')(obs)
    padded_next_obs = transforms.Pad((4, 4, 4, 4), padding_mode='edge')(next_obs)
    mag_x = np.random.randint(8)
    mag_y = np.random.randint(8)
    obs = padded_obs[:, :, mag_x:mag_x+heightmap_size, mag_y:mag_y+heightmap_size]
    next_obs = padded_next_obs[:, :, mag_x:mag_x+heightmap_size, mag_y:mag_y+heightmap_size]
    return obs, action, reward, next_obs, done

def perturb(current_image, next_image, dxy1, dxy2, set_theta_zero=False, set_trans_zero=False):
    image_size = current_image.shape[-2:]
    
    # Compute random rigid transform.
    theta, trans, pivot = get_random_image_transform_params(image_size)
    if set_theta_zero:
        theta = 0.
    if set_trans_zero:
        trans = [0., 0.]
    transform = get_image_transform(theta, trans, pivot)
    transform_params = theta, trans, pivot

    rot = np.array([[np.cos(theta), -np.sin(theta)], 
                    [np.sin(theta), np.cos(theta)]])
    rotated_dxy1 = rot.dot(dxy1)
    rotated_dxy1 = np.clip(rotated_dxy1, -1, 1)
    
    rotated_dxy2 = rot.dot(dxy2)
    rotated_dxy2 = np.clip(rotated_dxy2, -1, 1)

    # Apply rigid transform to image and pixel labels.
    if current_image.shape[0] == 1:
        current_image = affine_transform(current_image, np.linalg.inv(transform), mode='nearest', order=1)
        if next_image is not None:
            next_image = affine_transform(next_image, np.linalg.inv(transform), mode='nearest', order=1)
    else:
        for i in range(current_image.shape[0]):
            current_image[i, :, :] = affine_transform(current_image[i, :, :], np.linalg.inv(transform), mode='nearest', order=1)
            if next_image is not None:
                next_image[i, :, :] = affine_transform(next_image[i, :, :], np.linalg.inv(transform), mode='nearest', order=1)
    return current_image, next_image, rotated_dxy1, rotated_dxy2, transform_params


def get_random_image_transform_params(image_size):
    theta = np.random.random() * 2*np.pi
    trans = np.random.randint(0, image_size[0]//10, 2) - image_size[0]//20
    pivot = (image_size[1] / 2, image_size[0] / 2)
    return theta, trans, pivot

def get_image_transform(theta, trans, pivot=(0, 0)):
    """Compute composite 2D rigid transformation matrix."""
    # Get 2D rigid transformation matrix that rotates an image by theta (in
    # radians) around pivot (in pixels) and translates by trans vector (in
    # pixels)
    pivot_t_image = np.array([[1., 0., -pivot[0]], 
                              [0., 1., -pivot[1]],
                              [0., 0., 1.]])
    image_t_pivot = np.array([[1., 0., pivot[0]], 
                              [0., 1., pivot[1]],
                              [0., 0., 1.]])
    transform = np.array([[np.cos(theta), -np.sin(theta), trans[0]],
                          [np.sin(theta), np.cos(theta), trans[1]], 
                          [0., 0., 1.]])
    return np.dot(image_t_pivot, np.dot(transform, pivot_t_image))

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype
        )
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)


def random_crop(imgs, output_size):
    """
    Vectorized way to do random crop using sliding windows
    and picking out random ones

    args:
        imgs, batch images with shape (B,C,H,W)
    """
    # batch size
    n = imgs.shape[0]
    img_size = imgs.shape[-1]
    crop_max = img_size - output_size
    imgs = np.transpose(imgs, (0, 2, 3, 1))
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    # creates all sliding windows combinations of size (output_size)
    windows = view_as_windows(
        imgs, (1, output_size, output_size, 1))[..., 0, :, :, 0]
    # selects a random window for each batch element
    cropped_imgs = windows[np.arange(n), w1, h1]
    return cropped_imgs


def center_crop_image(image, output_size):
    # print('input image shape:', image.shape)
    if image.shape[0] ==1:
        image = image[0]
    h, w = image.shape[1:]
    new_h, new_w = output_size, output_size

    top = (h - new_h) // 2
    left = (w - new_w) // 2

    image = image[:, top:top + new_h, left:left + new_w]
    # print('output image shape:', image.shape)
    return image
