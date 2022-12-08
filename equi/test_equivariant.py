from equi.encoder import PixelEncoderEquivariant
from equi.equi_agent import ActorEquivariant, CriticEquivariant

import escnn
from escnn import gspaces

import torch
import numpy as np

def main(message, x, act, obs_shape, num_filters, feature_dim, N):
    if message == 'Test equivariant encoder':
        encoder = PixelEncoderEquivariant(obs_shape=obs_shape, feature_dim=feature_dim, N=N, num_filters=num_filters)
        field = escnn.nn.FieldType(act, obs_shape[0]*[act.trivial_repr])
        y = encoder(escnn.nn.GeometricTensor(x, field))
        print('='*50)
        print('Test equivariant encoder')
        print('='*50)

        x = field(x)

        for i, g in enumerate(act.testing_elements):
            print(i, g)
            # transform y by g
            y_tr = y.transform(g)

            # transform x by g
            x_tr = x.transform(g).tensor
            y_ = encoder(escnn.nn.GeometricTensor(x_tr, field))

            print(y_.tensor.reshape(1, -1))
            print(y_tr.tensor.reshape(1, -1))
            assert torch.allclose(y_.tensor, y_tr.tensor, atol=1e-2)
            print('OK')
            print('-'*50)

    elif message == 'Test equivariant actor':
        print('='*50)
        print('Test equivariant actor')
        print('='*50)
        actor = ActorEquivariant(obs_shape=obs_shape, action_shape=(8,), hidden_dim=feature_dim, encoder_type='pixel-equivariant', encoder_feature_dim=feature_dim, log_std_min=-20, log_std_max=2, num_layers=1, num_filters=16, N=N)
        out, _, _, ls = actor(x)
        x = escnn.nn.FieldType(act, 3*[act.trivial_repr])(x)
        out = torch.cat([out[:, 0:1], out[:, 2:3], out[:, 4:5], out[:, 6:7], out[:, 1:2], out[:, 3:4], out[:, 5:6], out[:, 7:8], ls], dim=1).reshape(1, -1, 1, 1)
        out_type = escnn.nn.FieldType(act, 2*[act.irrep(1)] + 12*[act.trivial_repr])

        out_ = out_type(out)
        for i, g in enumerate(act.testing_elements):
            print(i, g)
            out_tr = out_.transform(g).tensor.reshape(1, -1)
            out_tr = torch.cat([out_tr[:, 0:1], out_tr[:, 4:5], out_tr[:, 1:2], out_tr[:, 5:6], out_tr[:, 2:3], out_tr[:, 6:7], out_tr[:, 3:4], out_tr[:, 7:8], out_tr[:, 8:]], dim=1)

            x_tr = x.transform(g)
            out_x_tr, _, _, ls_x_tr = actor(x_tr.tensor)
            out_x_tr_ = torch.cat([out_x_tr, ls_x_tr], dim=1)

            print(out_tr)
            print(out_x_tr_)
            assert torch.allclose(out_x_tr_, out_tr, atol=1e-2)
            print('OK')
            print('-'*50)
    elif message == 'Test equivariant critic':
        print('='*50)
        print('Test equivariant critic')
        print('='*50)
        action = torch.randn(1, 8)
        critic = CriticEquivariant(obs_shape=obs_shape, action_shape=(8,), hidden_dim=feature_dim, encoder_type='pixel-equivariant', encoder_feature_dim=feature_dim, num_layers=1, num_filters=16, N=N)

        out1, out2 = critic(x, action)
        x = escnn.nn.FieldType(act, 3*[act.trivial_repr])(x)
        action = torch.cat([action[:, 1:2], action[:, 3:4], action[:, 5:6], action[:, 7:8], action[:, 0:1], action[:, 2:3], action[:, 4:5], action[:, 6:7]], dim=1).reshape(1, 8, 1, 1)
        action = escnn.nn.FieldType(act, 4 * [act.trivial_repr] + 2*[act.irrep(1)])(action)

        for i, g in enumerate(act.testing_elements):
            print(i, g)
            out1_tr, out2_tr = out1, out2
            x_tr = x.transform(g)
            a_tr = action.transform(g).tensor
            a_tr_ = torch.cat([a_tr[:, 4:5, :, :], a_tr[:, 0:1, :, :], a_tr[:, 5:6, :, :], a_tr[:, 1:2, :, :], a_tr[:, 6:7, :, :], a_tr[:, 2:3, :, :], a_tr[:, 7:8, :, :], a_tr[:, 3:4, :, :]], dim=1).reshape(1, 8)
            out1_, out2_ = critic(x_tr.tensor, a_tr_)
            print(out1_)
            print(out1_tr)
            assert torch.allclose(out1_, out1_tr, atol=1e-1)
            assert torch.allclose(out2_, out2_tr, atol=1e-1)
            print('OK')
            print('-'*50)

    else:
        print('Wrong message')

if __name__ == '__main__':
    N = 8
    x = torch.randint(0, 255, (1, 3, 128, 128), dtype=torch.float32) / 255.0
    act = gspaces.rot2dOnR2(N)
    obs_shape = [3, 128, 128]
    num_filters = 16
    feature_dim = 256
    main('Test equivariant encoder', x, act, obs_shape, num_filters, feature_dim, N)
    main('Test equivariant actor', x, act, obs_shape, num_filters, feature_dim, N)
    main('Test equivariant critic', x, act, obs_shape, num_filters, feature_dim, N)

