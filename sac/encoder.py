import torch
import torch.nn as nn


import escnn
from escnn import gspaces

def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


# for 100 x 100 inputs
OUT_DIM_100 = {4: 43}
OUT_DIM_128 = {4: 57}
# for 84 x 84 inputs
OUT_DIM = {2: 39, 4: 35, 6: 31}
# for 64 x 64 inputs
OUT_DIM_64 = {2: 29, 4: 25, 6: 21}


class ConvolutionEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim=256, num_layers=4, num_filters=32, output_logits=False, N=4):
        super().__init__()
        assert len(obs_shape) == 3
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim

        self.convs1 = torch.nn.Sequential(
            nn.Conv2d(3,16,kernel_size = 3,padding = 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
        )
            # 64 x 64
        self.convs2 = torch.nn.Sequential(
            nn.Conv2d(16,32,kernel_size = 3,padding = 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            )

            # 32 x 32
        self.convs3 = torch.nn.Sequential(
            nn.Conv2d(32,64,kernel_size = 3,padding = 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            )

            # 16 x 16
        self.convs4 = torch.nn.Sequential(
            # import ipdb;ipdb.set_trace(),
            nn.Conv2d(64,128,kernel_size = 3,padding = 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            nn.Conv2d(128,128,kernel_size = 3,padding = 1),
            nn.ReLU(inplace = True),

            nn.Conv2d(128,128,kernel_size = 3,padding = 0),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            )
        
        self.convs5 = torch.nn.Sequential(
            # import ipdb;ipdb.set_trace(),
            nn.Conv2d(128,self.feature_dim,kernel_size = 3,padding = 0),
            nn.ReLU(inplace = True),
            # nn.MaxPool2d(kernel_size = 2, stride = 2),
            )


    def forward(self, geo, detach=False):
        # print(self.feature_dim)
        gg = self.convs1(geo)
        gg = self.convs2(gg)
        gg = self.convs3(gg)
        gg = self.convs4(gg)
        gg = self.convs5(gg)
        if detach:
            gg = gg.detach()
        gg= gg.view(gg.size(0), -1)
        return gg



class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32,output_logits=False):
        super().__init__()
        # print(obs_shape)
        # print(len(obs_shape))
        # exit()
        assert len(obs_shape) == 3
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        if obs_shape[-1] == 64:
            out_dim =OUT_DIM_64[num_layers]
        elif obs_shape[-1] == 84:
            out_dim = OUT_DIM[num_layers]
        elif obs_shape[-1] == 100:
            out_dim = OUT_DIM_100[num_layers]
        elif obs_shape[-1] == 128:
            out_dim = OUT_DIM_128[num_layers]
        else:
            raise NotImplementedError
        # print("out_dim",out_dim)
        # print("can tinh",num_filters * out_dim * out_dim)
        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()
        self.output_logits = output_logits

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_conv(self, obs):
        # obs = torch.randn([128,3,128,128]).to(torch.float32).to(obs.device)
        obs = obs / 255.
        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        h = conv.view(conv.size(0), -1)
        # print(h.shape)
        # exit()
        return h

    def forward(self, obs, detach=False):
        assert detach == False
        # import ipdb;ipdb.set_trace()
        # print("obs",obs.shape)
        # obs = torch.randn([1,3,128,128]).to(torch.float32).to(obs.device)
        h = self.forward_conv(obs)


        if detach:
            h = h.detach()
        h_fc = self.fc(h)

        self.outputs['fc'] = h_fc

        h_norm = self.ln(h_fc)
        self.outputs['ln'] = h_norm

        if self.output_logits:
            out = h_norm
        else:
            out = torch.tanh(h_norm)
            self.outputs['tanh'] = out
        # print("out",out.shape)
        # exit()
        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param('train_encoder/conv%s' % (i + 1), self.convs[i], step)
        L.log_param('train_encoder/fc', self.fc, step)
        L.log_param('train_encoder/ln', self.ln, step)


class IdentityEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters,*args):
        super().__init__()

        assert len(obs_shape) == 1
        self.feature_dim = obs_shape[0]

    def forward(self, obs, detach=False):
        return obs

    def copy_conv_weights_from(self, source):
        pass

    def log(self, L, step, log_freq):
        pass


# _AVAILABLE_ENCODERS = {'pixel': PixelEncoder, 'identity': IdentityEncoder, 'pixel-equivariant': PixelEncoderEquivariant}
_AVAILABLE_ENCODERS = {'pixel': ConvolutionEncoder, 'identity': IdentityEncoder}

def make_encoder(
    encoder_type, obs_shape, feature_dim, num_layers, num_filters, output_logits=False, N=4
):
    assert encoder_type in _AVAILABLE_ENCODERS
    if encoder_type != 'pixel-equivariant':
        return _AVAILABLE_ENCODERS[encoder_type](
            obs_shape, feature_dim, num_layers, num_filters, output_logits
        )
    return _AVAILABLE_ENCODERS[encoder_type](
        obs_shape, feature_dim, num_layers, num_filters, output_logits, N
    )