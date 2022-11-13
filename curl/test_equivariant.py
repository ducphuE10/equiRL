from curl.encoder import PixelEncoderEquivariant
from curl.curl_sac import ActorEquivariant, CriticEquivariant

import escnn
from escnn import gspaces

import torch
import numpy as np

N = 8
x = torch.randint(0, 255, (1, 3, 128, 128), dtype=torch.float32) / 255.0
act = gspaces.rot2dOnR2(N)
obs_shape = [3, 128, 128]
num_filters = 32
feature_dim = 256

# encoder = PixelEncoderEquivariant(obs_shape=obs_shape, feature_dim=feature_dim, N=N, num_filters=num_filters)
# y = encoder(escnn.nn.GeometricTensor(x, escnn.nn.FieldType(act, 3*[act.trivial_repr])))
# print('Test equivariant encoder')
# print('='*50)

# x = escnn.nn.FieldType(act, 3*[act.trivial_repr])(x)

# for i, g in enumerate(act.testing_elements):
#     print(i, g)
#     y_tr = y.transform(g)

#     x_tr = x.transform(g)
#     x_90 = x_tr.tensor
#     y_ = encoder(escnn.nn.GeometricTensor(x_90, escnn.nn.FieldType(act, 3*[act.trivial_repr])))

#     print(y_.tensor.reshape(1, -1))
#     print(y_tr.tensor.reshape(1, -1))
#     assert torch.allclose(y_.tensor, y_tr.tensor, atol=1)
#     print('OK')
#     print('-'*50)

# print('Test equivariant actor')
# actor = ActorEquivariant(obs_shape=obs_shape, action_shape=(8,), hidden_dim=feature_dim, encoder_type='pixel-equivariant', encoder_feature_dim=feature_dim, log_std_min=-20, log_std_max=2, num_layers=1, num_filters=16, N=N)
# out = actor(x)
# x = escnn.nn.FieldType(act, 3*[act.trivial_repr])(x)

# for i, g in enumerate(act.testing_elements):
#     print(i, g)
#     out_tr = out.transform(g)

#     # obs_90 = torch.rot90(obs, i, [2, 3])
#     x_tr = x.transform(g)
#     out_ = actor(x_tr.tensor)

#     assert torch.allclose(out_.tensor, out_tr.tensor, atol=1e-3)
#     print('OK')
#     print('-'*50)

print('Test equivariant critic')
action = torch.randn(1, 8)
critic = CriticEquivariant(obs_shape=obs_shape, action_shape=(8,), hidden_dim=feature_dim, encoder_type='pixel-equivariant', encoder_feature_dim=feature_dim, num_layers=1, num_filters=16, N=4)

out1, out2 = critic(x, action)
x = escnn.nn.FieldType(act, 3*[act.trivial_repr])(x)
action = torch.cat([action[:, 0:1], action[:, 2:3], action[:, 4:5], action[:, 6:7], action[:, 1:2], action[:, 3:4], action[:, 5:6], action[:, 7:8]], dim=1).reshape(1, 8, 1, 1)
action = escnn.nn.FieldType(act, 4 * [act.trivial_repr] + 2*[act.irrep(1)])(action)

for i, g in enumerate(act.testing_elements):
    # import ipdb; ipdb.set_trace()
    print(i, g)
    out1_tr, out2_tr = out1, out2

    # obs_90 = torch.rot90(obs, i, [2, 3])
    x_tr = x.transform(g)
    a_tr = action.transform(g).tensor
    a_tr_ = torch.cat([a_tr[:, 0:1, :, :], a_tr[:, 4:5, :, :], a_tr[:, 1:2, :, :], a_tr[:, 5:6, :, :], a_tr[:, 2:3, :, :], a_tr[:, 6:7, :, :], a_tr[:, 3:4, :, :], a_tr[:, 7:8, :, :]], dim=1).reshape(1, 8)
    # rot = (90*i / 360) * 2 * np.pi
    # mtr = torch.tensor([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]]).float()
    # act_equi_1 = torch.Tensor([action[0], action[2]])
    # act_equi_2 = torch.Tensor([action[4], action[6]])
    # act_equi_1 = torch.matmul(mtr, act_equi_1)
    # act_equi_2 = torch.matmul(mtr, act_equi_2)
    # action_rot = torch.Tensor([act_equi_1[0], action[1:2], act_equi_1[1], action[3:4], act_equi_2[0], action[5:6], act_equi_2[1], action[7:8]])
    out1_, out2_ = critic(x_tr.tensor, a_tr_)

    assert torch.allclose(out1_, out1_tr, atol=1)
    assert torch.allclose(out2_, out2_tr, atol=1)
    print('OK')


