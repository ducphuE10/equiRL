from curl.encoder import PixelEncoderEquivariant
from curl.curl_sac import ActorEquivariant, CriticEquivariant

import escnn
from escnn import gspaces

import torch

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
print(out1, out2)

