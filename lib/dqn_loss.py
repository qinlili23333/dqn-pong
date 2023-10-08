import torch
import torch.nn as nn

def calc_loss(batch, net, tgt_net, GAMMA):
    states_v, actions_v, rewards_v, dones_v, next_states_v = batch

    Q_s = net.forward(states_v.type(torch.float))
    state_action_values = Q_s.gather(1, actions_v.type(torch.int64).unsqueeze(-1)).squeeze(-1)
    next_state_values = tgt_net.forward(next_states_v.type(torch.float)).max(1)[0]

    expected_state_action_values = rewards_v + next_state_values.detach() * GAMMA * (1 - dones_v)
    return nn.MSELoss()(state_action_values, expected_state_action_values)

