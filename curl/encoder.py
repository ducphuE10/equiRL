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

# for 84 x 84 inputs
OUT_DIM = {2: 39, 4: 35, 6: 31}
# for 64 x 64 inputs
OUT_DIM_64 = {2: 29, 4: 25, 6: 21}


class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32,output_logits=False):
        super().__init__()

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
        else:
            raise NotImplementedError
        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()
        self.output_logits = output_logits

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_conv(self, obs):
        obs = obs / 255.
        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
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

class PixelEncoderEquivariant(nn.Module):
    """Equivariant convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim=512, num_layers=4, num_filters=32, output_logits=False, N=4):
        super().__init__()

        assert len(obs_shape) == 3
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim

        self.act = gspaces.rot2dOnR2(N)

        self.convs1 = torch.nn.Sequential(
            # 100 x100
            escnn.nn.R2Conv(escnn.nn.FieldType(self.act, self.obs_shape[0]*[self.act.trivial_repr]),
                            escnn.nn.FieldType(self.act, num_filters*[self.act.regular_repr]),
                            kernel_size=3, padding=1, initialize=True),
            escnn.nn.ReLU(escnn.nn.FieldType(self.act, num_filters*[self.act.regular_repr]), inplace=True),
            escnn.nn.PointwiseMaxPool(escnn.nn.FieldType(self.act, num_filters*[self.act.regular_repr]), 2))

            # 50 x 50
        self.convs2 = torch.nn.Sequential(
            escnn.nn.R2Conv(escnn.nn.FieldType(self.act, num_filters*[self.act.regular_repr]),
                            escnn.nn.FieldType(self.act, 2*num_filters*[self.act.regular_repr]),
                            kernel_size=3, padding=1, initialize=True),
            escnn.nn.ReLU(escnn.nn.FieldType(self.act, 2*num_filters*[self.act.regular_repr]), inplace=True),
            escnn.nn.PointwiseMaxPool(escnn.nn.FieldType(self.act, 2*num_filters*[self.act.regular_repr]), 2))

            # 25 x 25
        self.convs3 = torch.nn.Sequential(
            escnn.nn.R2Conv(escnn.nn.FieldType(self.act, 2*num_filters*[self.act.regular_repr]),
                            escnn.nn.FieldType(self.act, 2*num_filters*[self.act.regular_repr]),
                            kernel_size=3, padding=1, initialize=True),
            escnn.nn.ReLU(escnn.nn.FieldType(self.act, 2*num_filters*[self.act.regular_repr]), inplace=True),
            escnn.nn.PointwiseMaxPool(escnn.nn.FieldType(self.act, 2*num_filters*[self.act.regular_repr]), 2))

            # 12 x 12
        self.convs4 = torch.nn.Sequential(
            escnn.nn.R2Conv(escnn.nn.FieldType(self.act, 2*num_filters*[self.act.regular_repr]),
                            escnn.nn.FieldType(self.act, 4*num_filters*[self.act.regular_repr]),
                            kernel_size=3, padding=0, initialize=True),
            escnn.nn.ReLU(escnn.nn.FieldType(self.act, 4*num_filters*[self.act.regular_repr]), inplace=True),
            escnn.nn.PointwiseMaxPool(escnn.nn.FieldType(self.act, 4*num_filters*[self.act.regular_repr]), 2),

            # 5 x 5
            escnn.nn.R2Conv(escnn.nn.FieldType(self.act, 4*num_filters*[self.act.regular_repr]),
                            escnn.nn.FieldType(self.act, 4*num_filters*[self.act.regular_repr]),
                            kernel_size=3, padding=0, initialize=True),
            escnn.nn.ReLU(escnn.nn.FieldType(self.act, 4*num_filters*[self.act.regular_repr]), inplace=True))

            # 3 x 3
        self.convs5 = torch.nn.Sequential(
            escnn.nn.R2Conv(escnn.nn.FieldType(self.act, 4*num_filters*[self.act.regular_repr]),
                            escnn.nn.FieldType(self.act, self.feature_dim*[self.act.regular_repr]),
                            kernel_size=3, padding=0, initialize=True),
            escnn.nn.ReLU(escnn.nn.FieldType(self.act, self.feature_dim*[self.act.regular_repr]), inplace=True))

            # 1 x 1

    def forward(self, geo):
        gg = self.convs1(geo)
        # print('1', gg.tensor.shape)
        gg = self.convs2(gg)
        # print('2', gg.tensor.shape)
        gg = self.convs3(gg)
        # print('3', gg.tensor.shape)
        gg = self.convs4(gg)
        # print('4', gg.tensor.shape)
        gg = self.convs5(gg)
        # print('5', gg.tensor.shape)
        return gg

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

# _AVAILABLE_ENCODERS = {'pixel': PixelEncoder, 'identity': IdentityEncoder,}
_AVAILABLE_ENCODERS = {'pixel': PixelEncoder, 'identity': IdentityEncoder, 'pixel-equivariant': PixelEncoderEquivariant}


def make_encoder(
    encoder_type, obs_shape, feature_dim, num_layers, num_filters, output_logits=False
):
    assert encoder_type in _AVAILABLE_ENCODERS
    return _AVAILABLE_ENCODERS[encoder_type](
        obs_shape, feature_dim, num_layers, num_filters, output_logits
    )

if __name__ == '__main__':
    encoder = PixelEncoderEquivariant(obs_shape=(3, 100, 100), feature_dim=128, num_filters=16, N=4)
    print(encoder)
    x = torch.rand(1, 3, 100, 100)
    act = gspaces.rot2dOnR2(N=4)
    y = encoder(escnn.nn.GeometricTensor(x, escnn.nn.FieldType(act, x.shape[1]*[act.trivial_repr])))
    print(y.shape)