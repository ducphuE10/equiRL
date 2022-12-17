import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

import escnn
from escnn import gspaces

from equi import utils
from equi.encoder import make_encoder
import wandb

LOG_FREQ = 10000

def get_optimizer_stats(optim):
    state_dict = optim.state_dict()['state']
    if len(state_dict) ==0:
        flattened_exp_avg = flattened_exp_avg_sq = 0.
        print('Warning: optimizer dict is empty!')
    else:
        # flattened_exp_avg = flattened_exp_avg_sq = 0.
        flattened_exp_avg = torch.cat([x['exp_avg'].flatten() for x in state_dict.values()]).cpu().numpy()
        flattened_exp_avg_sq = torch.cat([x['exp_avg_sq'].flatten() for x in state_dict.values()]).cpu().numpy()
    return {
        'exp_avg_mean': np.mean(flattened_exp_avg),
        'exp_avg_std': np.std(flattened_exp_avg),
        'exp_avg_min': np.min(flattened_exp_avg),
        'exp_avg_max': np.max(flattened_exp_avg),
        'exp_avg_sq_mean': np.mean(flattened_exp_avg_sq),
        'exp_avg_sq_std': np.std(flattened_exp_avg_sq),
        'exp_avg_sq_min': np.min(flattened_exp_avg_sq),
        'exp_avg_sq_max': np.max(flattened_exp_avg_sq),
    }


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class Actor(nn.Module):
    """MLP actor network."""
    def __init__(
      self, obs_shape, action_shape, hidden_dim, encoder_type,
      encoder_feature_dim, log_std_min, log_std_max, num_layers, num_filters
    ):
        
        super().__init__()

        assert encoder_type == 'pixel'
        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters, output_logits=True
        )

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_shape[0])
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(
      self, obs, compute_pi=True, compute_log_pi=True, detach_encoder=False
    ):
        obs = self.encoder(obs, detach=detach_encoder)
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
          self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        self.outputs['mu'] = mu
        self.outputs['std'] = log_std.exp()

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std

    def forward_from_feature(self, obs, compute_pi=False, compute_log_pi=False):
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
          self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        self.outputs['mu'] = mu
        self.outputs['std'] = log_std.exp()

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, log_std

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_actor/%s_hist' % k, v, step)

        L.log_param('train_actor/fc1', self.trunk[0], step)
        L.log_param('train_actor/fc2', self.trunk[2], step)
        L.log_param('train_actor/fc3', self.trunk[4], step)

class ActorEquivariant(nn.Module):
    """Equivariant actor network."""
    def __init__(self, obs_shape, action_shape, hidden_dim, encoder_type,
                 encoder_feature_dim, log_std_min, log_std_max, num_layers,
                 num_filters, N):
        super().__init__()

        print(f'===================================== Equivariant Actor with C{N}=====================================')

        assert encoder_feature_dim == hidden_dim
        self.act = gspaces.rot2dOnR2(N)

        self.action_shape = action_shape
        self.obs_shape = obs_shape
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers, num_filters, output_logits=False, N=N
        )

        self.conv = nn.Sequential(
            escnn.nn.R2Conv(escnn.nn.FieldType(self.act, hidden_dim*[self.act.regular_repr]),
                            escnn.nn.FieldType(self.act, 2*[self.act.irrep(1)] + (self.action_shape[0]*2 -4)*[self.act.trivial_repr]), 
                            kernel_size=1, padding=0, initialize=True)
        )

    def forward(self, obs, compute_pi=True, compute_log_pi=True, detach_encoder=False):
        obs = obs / 255.0
        obs_geo = escnn.nn.GeometricTensor(obs, escnn.nn.FieldType(self.act, self.obs_shape[0]*[self.act.trivial_repr]))
        conv_out = self.conv(self.encoder(obs_geo, detach_encoder)).tensor.reshape(obs.shape[0], -1)
        dxy = conv_out[:, :4]
        inv_act = conv_out[:, 4:self.action_shape[0]]
        mean = torch.cat((dxy[:, 0:1], inv_act[:, 0:1], dxy[:, 1:2], inv_act[:, 1:2], dxy[:, 2:3], inv_act[:, 2:3], dxy[:, 3:4], inv_act[:, 3:4]), dim=1)
        log_std = conv_out[:, self.action_shape[0]:]

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
          self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mean)
            pi = mean + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mean, pi, log_pi = squash(mean, pi, log_pi)

        return mean, pi, log_pi, log_std


class ActorEquivariant_1(nn.Module):
    """Equivariant actor network."""
    def __init__(self, obs_shape, action_shape, hidden_dim, encoder_type,
                 encoder_feature_dim, log_std_min, log_std_max, num_layers,
                 num_filters, N):
        super().__init__()

        print(f'===================================== Equivariant Actor 1 with C{N}=====================================')

        assert encoder_feature_dim == hidden_dim
        self.act = gspaces.rot2dOnR2(N)

        self.action_shape = action_shape
        self.obs_shape = obs_shape
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers, num_filters, output_logits=False, N=N
        )

        self.conv = nn.Sequential(
            escnn.nn.R2Conv(escnn.nn.FieldType(self.act, hidden_dim*[self.act.regular_repr]),
                            escnn.nn.FieldType(self.act, 1*[self.act.irrep(1)] + (self.action_shape[0]*2 -6)*[self.act.trivial_repr]), 
                            kernel_size=1, padding=0, initialize=True)
        )

    def forward(self, obs, compute_pi=True, compute_log_pi=True, detach_encoder=False):
        obs = obs / 255.0
        obs_geo = escnn.nn.GeometricTensor(obs, escnn.nn.FieldType(self.act, self.obs_shape[0]*[self.act.trivial_repr]))
        conv_out = self.conv(self.encoder(obs_geo, detach_encoder)).tensor.reshape(obs.shape[0], -1)
        mean = conv_out[:, :self.action_shape[0]-2]
        log_std = conv_out[:, self.action_shape[0]-2:]

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
          self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mean)
            pi = mean + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mean, pi, log_pi = squash(mean, pi, log_pi)

        return mean, pi, log_pi, log_std

class QFunction(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, action):
        if obs.size(0) != action.size(0):
            print(obs.size)
            print(action.size)
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=1)
        return self.trunk(obs_action)

class Critic(nn.Module):
    """Critic network, employes two q-functions."""

    def __init__(
      self, obs_shape, action_shape, hidden_dim, encoder_type,
      encoder_feature_dim, num_layers, num_filters
    ):
        super().__init__()

        self.encoder1 = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters, output_logits=True
        )

        self.encoder2 = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters, output_logits=True
        )

        self.Q1 = QFunction(
            self.encoder1.feature_dim, action_shape[0], hidden_dim
        )
        self.Q2 = QFunction(
            self.encoder2.feature_dim, action_shape[0], hidden_dim
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, action, detach_encoder=False):
        # detach_encoder allows to stop gradient propogation to encoder
        obs1 = self.encoder1(obs, detach=detach_encoder)
        obs2 = self.encoder2(obs, detach=detach_encoder)
        q1 = self.Q1(obs1, action)
        q2 = self.Q2(obs2, action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def forward_from_feature(self, feature, action):
        # detach_encoder allows to stop gradient propogation to encoder
        q1 = self.Q1(feature, action)
        q2 = self.Q2(feature, action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        self.encoder1.log(L, step, log_freq)
        self.encoder2.log(L, step, log_freq)

        for k, v in self.outputs.items():
            L.log_histogram('train_critic/%s_hist' % k, v, step)

        for i in range(3):
            L.log_param('train_critic/q1_fc%d' % i, self.Q1.trunk[i * 2], step)
            L.log_param('train_critic/q2_fc%d' % i, self.Q2.trunk[i * 2], step)

class CriticEquivariant(nn.Module):
    """Equivariant critic network, employes two q-functions."""
    def __init__(
      self, obs_shape, action_shape, hidden_dim, encoder_type,
      encoder_feature_dim, num_layers, num_filters, N
    ):
        super().__init__()
        assert hidden_dim == encoder_feature_dim
        print(f'===================================== Equivariant Critic with C{N} =====================================')

        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters, output_logits=False, N=N
        )
    
        self.act = gspaces.rot2dOnR2(N)
        self.action_shape = action_shape
        self.obs_shape = obs_shape
        self.hidden_dim = hidden_dim

        self.Q1 = nn.Sequential(
            escnn.nn.R2Conv(escnn.nn.FieldType(self.act, hidden_dim*[self.act.regular_repr] + (self.action_shape[0] - 4)*[self.act.trivial_repr] + 2*[self.act.irrep(1)]),
                            escnn.nn.FieldType(self.act, hidden_dim*[self.act.regular_repr]), 
                            kernel_size=1, padding=0, initialize=True),
            escnn.nn.ReLU(escnn.nn.FieldType(self.act, hidden_dim*[self.act.regular_repr]), inplace=True),
            escnn.nn.GroupPooling(escnn.nn.FieldType(self.act, hidden_dim*[self.act.regular_repr])),
            escnn.nn.R2Conv(escnn.nn.FieldType(self.act, hidden_dim*[self.act.trivial_repr]),
                            escnn.nn.FieldType(self.act, 1*[self.act.trivial_repr]), 
                            kernel_size=1, padding=0, initialize=True),
        )

        self.Q2 = nn.Sequential(
            escnn.nn.R2Conv(escnn.nn.FieldType(self.act, hidden_dim*[self.act.regular_repr] + (self.action_shape[0] - 4)*[self.act.trivial_repr] + 2*[self.act.irrep(1)]),
                            escnn.nn.FieldType(self.act, hidden_dim*[self.act.regular_repr]), 
                            kernel_size=1, padding=0, initialize=True),
            escnn.nn.ReLU(escnn.nn.FieldType(self.act, hidden_dim*[self.act.regular_repr]), inplace=True),
            escnn.nn.GroupPooling(escnn.nn.FieldType(self.act, hidden_dim*[self.act.regular_repr])),
            escnn.nn.R2Conv(escnn.nn.FieldType(self.act, hidden_dim*[self.act.trivial_repr]),
                            escnn.nn.FieldType(self.act, 1*[self.act.trivial_repr]), 
                            kernel_size=1, padding=0, initialize=True),
        )
    
    def forward(self, obs, action, detach_encoder=False):
        obs = obs / 255.0
        batch_size = obs.shape[0]
        action = action.reshape(batch_size, -1)
        obs_geo = escnn.nn.GeometricTensor(obs, escnn.nn.FieldType(self.act, self.obs_shape[0]*[self.act.trivial_repr]))
        conv_out = self.encoder(obs_geo, detach=detach_encoder)   
        dxy = torch.cat([action[:, 0:1], action[:, 2:3], action[:, 4:5], action[:, 6:7]], dim=1).reshape(batch_size, 4, 1, 1)
        inv_act = torch.cat([action[:, 1:2], action[:, 3:4], action[:, 5:6], action[:, 7:8]], dim=1).reshape(batch_size, 4, 1, 1)
        cat = torch.cat((conv_out.tensor, inv_act, dxy), dim=1)
        cat_geo = escnn.nn.GeometricTensor(cat, escnn.nn.FieldType(self.act, self.hidden_dim *[self.act.regular_repr] + (self.action_shape[0] - 4)*[self.act.trivial_repr] + 2*[self.act.irrep(1)]))
        q1 = self.Q1(cat_geo).tensor.reshape(batch_size, 1)
        q2 = self.Q2(cat_geo).tensor.reshape(batch_size, 1)
        return q1, q2

class CriticEquivariant_1(nn.Module):
    """Equivariant critic network, employes two q-functions."""
    def __init__(
      self, obs_shape, action_shape, hidden_dim, encoder_type,
      encoder_feature_dim, num_layers, num_filters, N
    ):
        super().__init__()
        assert hidden_dim == encoder_feature_dim
        print(f'===================================== Equivariant Critic1 with C{N} =====================================')

        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters, output_logits=False, N=N
        )
    
        self.act = gspaces.rot2dOnR2(N)
        self.action_shape = action_shape
        self.obs_shape = obs_shape
        self.hidden_dim = hidden_dim

        self.Q1 = nn.Sequential(
            escnn.nn.R2Conv(escnn.nn.FieldType(self.act, hidden_dim*[self.act.regular_repr] + 1*[self.act.irrep(1)] + (self.action_shape[0] - 4)*[self.act.trivial_repr]),
                            escnn.nn.FieldType(self.act, hidden_dim*[self.act.regular_repr]), 
                            kernel_size=1, padding=0, initialize=True),
            escnn.nn.ReLU(escnn.nn.FieldType(self.act, hidden_dim*[self.act.regular_repr]), inplace=True),
            escnn.nn.GroupPooling(escnn.nn.FieldType(self.act, hidden_dim*[self.act.regular_repr])),
            escnn.nn.R2Conv(escnn.nn.FieldType(self.act, hidden_dim*[self.act.trivial_repr]),
                            escnn.nn.FieldType(self.act, 1*[self.act.trivial_repr]), 
                            kernel_size=1, padding=0, initialize=True),
        )

        self.Q2 = nn.Sequential(
            escnn.nn.R2Conv(escnn.nn.FieldType(self.act, hidden_dim*[self.act.regular_repr] + 1*[self.act.irrep(1)] + (self.action_shape[0] - 4)*[self.act.trivial_repr]),
                            escnn.nn.FieldType(self.act, hidden_dim*[self.act.regular_repr]), 
                            kernel_size=1, padding=0, initialize=True),
            escnn.nn.ReLU(escnn.nn.FieldType(self.act, hidden_dim*[self.act.regular_repr]), inplace=True),
            escnn.nn.GroupPooling(escnn.nn.FieldType(self.act, hidden_dim*[self.act.regular_repr])),
            escnn.nn.R2Conv(escnn.nn.FieldType(self.act, hidden_dim*[self.act.trivial_repr]),
                            escnn.nn.FieldType(self.act, 1*[self.act.trivial_repr]),
                            kernel_size=1, padding=0, initialize=True),
        )
        
    def forward(self, obs, action, detach_encoder=False):
        obs = obs / 255.0
        batch_size = obs.shape[0]
        action = action.reshape(batch_size, -1, 1, 1)
        obs_geo = escnn.nn.GeometricTensor(obs, escnn.nn.FieldType(self.act, self.obs_shape[0]*[self.act.trivial_repr]))
        conv_out = self.encoder(obs_geo, detach=detach_encoder)
        cat = torch.cat([conv_out.tensor, action], dim=1)
        cat_geo = escnn.nn.GeometricTensor(cat, escnn.nn.FieldType(self.act, self.hidden_dim *[self.act.regular_repr] + 1*[self.act.irrep(1)] + (self.action_shape[0] - 4)*[self.act.trivial_repr]))
        q1 = self.Q1(cat_geo).tensor.reshape(batch_size, 1)
        q2 = self.Q2(cat_geo).tensor.reshape(batch_size, 1)
        return q1, q2

class SacAgent(object):
    def __init__(
      self,
      obs_shape,
      action_shape,
      device,
      args,
      hidden_dim=256,
      discount=0.99,
      init_temperature=0.01,
      alpha_lr=1e-3,
      alpha_beta=0.9,
      alpha_fixed=False,
      actor_lr=1e-3,
      actor_beta=0.9,
      actor_log_std_min=-10,
      actor_log_std_max=2,
      actor_update_freq=2,
      critic_lr=1e-3,
      critic_beta=0.9,
      critic_tau=0.005,
      critic_target_update_freq=2,
      encoder_type='identity',
      encoder_feature_dim=50,
      encoder_lr=1e-3,
      encoder_tau=0.005,
      num_layers=4,
      num_filters=32,
      cpc_update_freq=1,
      log_interval=100,
      detach_encoder=False,
      curl_latent_dim=128,
      num_rotations=8

    ):
        self.args = args
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.cpc_update_freq = cpc_update_freq
        self.log_interval = log_interval
        self.image_size = obs_shape[-1]
        self.curl_latent_dim = curl_latent_dim
        self.detach_encoder = detach_encoder
        self.encoder_type = encoder_type
        self.alpha_fixed = alpha_fixed
        
        # build equivariant actor model
        # self.actor = ActorEquivariant(
        #     obs_shape, action_shape, hidden_dim, 'pixel-equivariant',
        #     encoder_feature_dim, actor_log_std_min, actor_log_std_max,
        #     num_layers, num_filters, num_rotations).to(device)

        self.actor = ActorEquivariant_1(
            obs_shape, action_shape, hidden_dim, 'pixel-equivariant',
            encoder_feature_dim, actor_log_std_min, actor_log_std_max,
            num_layers, num_filters, num_rotations).to(device)


        # build equivariant critic model
        # self.critic = CriticEquivariant(
        #     obs_shape, action_shape, hidden_dim, 'pixel-equivariant',
        #     encoder_feature_dim, num_layers, num_filters, num_rotations).to(device)
        self.critic = CriticEquivariant_1(
            obs_shape, action_shape, hidden_dim, 'pixel-equivariant',
            encoder_feature_dim, num_layers, num_filters, num_rotations).to(device)

        # build equivariant encoder model
        # self.critic_target = CriticEquivariant(
        #     obs_shape, action_shape, hidden_dim, 'pixel-equivariant',
        #     encoder_feature_dim, num_layers, num_filters, num_rotations).to(device)
        self.critic_target = CriticEquivariant_1(
            obs_shape, action_shape, hidden_dim, 'pixel-equivariant',
            encoder_feature_dim, num_layers, num_filters, num_rotations).to(device)

        # copy critic parameters to critic target
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Tie encoder between actor and critic
        self.actor.encoder.load_state_dict(self.critic.encoder.state_dict())
        
        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape) + 2

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        if self.args.lr_decay is not None:
            # Actor is halved due to delayed update
            self.actor_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.actor_optimizer, milestones=np.arange(15, 150, 15) * 5000, gamma=0.5)
            self.critic_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.critic_optimizer, milestones=np.arange(15, 150, 15) * 10000, gamma=0.5)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        # import ipdb;ipdb.set_trace()
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs):
        with torch.no_grad():
            if not isinstance(obs, torch.Tensor):
                obs = torch.from_numpy(obs)
            obs = obs.to(torch.float32).to(self.device)
            obs = obs.unsqueeze(0)
            mu, _, _, _ = self.actor(obs, compute_pi=False, compute_log_pi=False)
            mu = utils.posprocess_action(mu)
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):     
        if obs.shape[0] == 1:
            obs = obs[0]

        with torch.no_grad():
            if not isinstance(obs, torch.Tensor):
                obs = torch.from_numpy(obs)
            obs = obs.to(torch.float32).to(self.device)
            obs = obs.unsqueeze(0)
            _, pi, _, _ = self.actor(obs, compute_log_pi=False)
            pi = utils.posprocess_action(pi)
            return pi.cpu().data.numpy().flatten()

    def update_critic(self, obs, action, reward, next_obs, not_done, L, step):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action, detach_encoder=self.detach_encoder)
        critic_loss = F.mse_loss(current_Q1,target_Q) + F.mse_loss(current_Q2, target_Q)
        if step % self.log_interval == 0:
            L.log('train_critic/loss', critic_loss, step)
            if self.args.wandb:
                wandb.log({'train_critic_loss': critic_loss}, step=step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.args.lr_decay is not None:
            self.critic_lr_scheduler.step()
            L.log('train/critic_lr', self.critic_optimizer.param_groups[0]['lr'], step)

        # self.critic.log(L, step)

    def update_actor_and_alpha(self, obs, L, step):
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.actor(obs)
        actor_Q1, actor_Q2 = self.critic(obs, pi)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        if step % self.log_interval == 0:
            L.log('train_actor/loss', actor_loss, step)
            if self.args.wandb:
                wandb.log({'train_actor_loss': actor_loss}, step=step)
        # entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.args.lr_decay is not None:
            self.actor_lr_scheduler.step()
            L.log('train/actor_lr', self.actor_optimizer.param_groups[0]['lr'], step)

        if not self.alpha_fixed:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()
            if step % self.log_interval == 0:
                L.log('train_alpha/loss', alpha_loss, step)
                L.log('train_alpha/value', self.alpha, step)
                if self.args.wandb:
                    wandb.log({'train_alpha_loss': alpha_loss}, step=step)
                    wandb.log({'train_alpha_value': self.alpha}, step=step)
                    wandb.log({'-log pi': -log_pi.mean()}, step=step)
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def update(self, replay_buffer, L, step):
        #sample from buffer
        # s_s = time.time()
        obs, action, reward, next_obs, not_done = replay_buffer.sample_proprio()
        action = utils.preprocess_action(action)
        # print('sample time', time.time() - s_s)

        if step % self.log_interval == 0:
            L.log('train/batch_reward', reward.mean(), step)
            if self.args.wandb:
                wandb.log({'train_batch_reward': reward.mean()}, step=step)

        #----Update----
        #Critic
        # s_c = time.time()
        self.update_critic(obs, action, reward, next_obs, not_done, L, step)
        # print('critic time', time.time() - s_c)
        #Actor
        if step % self.actor_update_freq == 0: #default actor_update_freq = 2
            # s_a = time.time()
            self.update_actor_and_alpha(obs, L, step)
            # print('actor time', time.time() - s_a)
        #soft update
        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)

    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )