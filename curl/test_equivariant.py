from curl.encoder import PixelEncoderEquivariant
from curl.curl_sac import ActorEquivariant, CriticEquivariant

import escnn
from escnn import gspaces

import torch
import numpy as np

x = torch.rand(1, 3, 128, 128)
act = gspaces.rot2dOnR2(4)
obs_shape = [3, 128, 128]
num_filters = 32
feature_dim = 256

# encoder = PixelEncoderEquivariant(obs_shape=obs_shape, feature_dim=feature_dim, N=4, num_filters=num_filters)
# y = encoder(escnn.nn.GeometricTensor(x, escnn.nn.FieldType(act, 3*[act.trivial_repr])))
# print('Test equivariant encoder')
# print('='*50)

# for i, g in enumerate(act.testing_elements):
#     print(i, g)
#     y_tr = y.transform(g)

#     x_90 = torch.rot90(x, i, [2, 3])
#     y_ = encoder(escnn.nn.GeometricTensor(x_90, escnn.nn.FieldType(act, 3*[act.trivial_repr])))

#     assert torch.allclose(y_.tensor, y_tr.tensor, atol=1e-5)
#     print('OK')

# print('Test equivariant actor')
# obs = torch.rand(1, 3, 128, 128)
# actor = ActorEquivariant(obs_shape=obs_shape, action_shape=(8,), hidden_dim=feature_dim, encoder_type='pixel-equivariant', encoder_feature_dim=feature_dim, log_std_min=-20, log_std_max=2, num_layers=1, num_filters=16, N=4)
# out = actor(obs)
# print(out)

# for i, g in enumerate(act.testing_elements):
#     print(i, g)
#     out_tr = out.transform(g)

#     obs_90 = torch.rot90(obs, i, [2, 3])
#     out_ = actor(obs_90)

#     assert torch.allclose(out_.tensor, out_tr.tensor, atol=1e-5)
#     print('OK')

print('Test equivariant critic')
obs = torch.randn(1, 3, 128, 128)
action = torch.randn(8)
critic = CriticEquivariant(obs_shape=obs_shape, action_shape=(8,), hidden_dim=feature_dim, encoder_type='pixel-equivariant', encoder_feature_dim=feature_dim, num_layers=1, num_filters=16, N=4)

out1, out2 = critic(obs, action)

for i, g in enumerate(act.testing_elements):
    # import ipdb; ipdb.set_trace()
    print(i, g)
    out1_tr, out2_tr = out1, out2

    obs_90 = torch.rot90(obs, i, [2, 3])
    rot = (90*i / 360) * 2 * np.pi
    mtr = torch.tensor([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]]).float()
    act_equi_1 = torch.Tensor([action[0], action[2]])
    act_equi_2 = torch.Tensor([action[4], action[6]])
    act_equi_1 = torch.matmul(mtr, act_equi_1)
    act_equi_2 = torch.matmul(mtr, act_equi_2)
    action_rot = torch.Tensor([act_equi_1[0], action[1:2], act_equi_1[1], action[3:4], act_equi_2[0], action[5:6], act_equi_2[1], action[7:8]])
    out1_, out2_ = critic(obs_90, action_rot)

    assert torch.allclose(out1_, out1_tr, atol=1e-5)
    assert torch.allclose(out2_, out2_tr, atol=1e-5)
    print('OK')

