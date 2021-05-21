from typing import Union, Tuple, List

import numpy as np
import torch
from numpy import ndarray
from ptan.experience import ExperienceFirstLast
from torch import nn, ByteTensor


class DQNModel(nn.Module):

    def __init__(self, view_shape, features_shape, number_of_actions):
        super(DQNModel, self).__init__()

        self.view_convolutional: nn.Sequential = nn.Sequential(
            nn.Conv2d(in_channels=view_shape[0],
                      out_channels=32,
                      kernel_size=3,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=12,
                      kernel_size=2,
                      padding=1),
            nn.ReLU())

        view_out_size: int = self._get_conv_out(view_shape)
        self.fully_connected: nn.Sequential = nn.Sequential(
            nn.Linear(in_features=view_out_size + features_shape[0], out_features=128),
            nn.ReLU(),
            nn.Linear(128, number_of_actions))

    def _get_conv_out(self, shape) -> int:
        output = self.view_convolutional(torch.zeros(1, *shape))
        return int(np.prod(output.size()))

    def forward(self, x):
        view_batch, features_batch = x
        batch_size = view_batch.size()[0]
        convolutional_output = self.view_convolutional(view_batch).view(batch_size, -1)
        fully_connected_input = torch.cat((convolutional_output, features_batch), dim=1)

        return self.fully_connected(fully_connected_input)


class MAgentPreprocessor:

    def __init__(self, device: Union[torch.device, str] = "cpu"):
        self.device: Union[torch.device, str] = device

    def __call__(self, batch: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[torch.tensor, torch.tensor]:
        view_arrays, feature_arrays = zip(*batch)
        view_tensor: torch.tensor = torch.tensor(view_arrays, dtype=torch.float32).to(self.device)
        feature_tensor: torch.tensor = torch.tensor(feature_arrays, dtype=torch.float32).to(self.device)

        return view_tensor, feature_tensor


def unpack_batch(batch: List[ExperienceFirstLast]) -> Tuple[List, ndarray, ndarray, ndarray, List]:
    states, actions, rewards, dones, last_states = [], [], [], [], []

    for experience in batch:
        states.append(experience.state)
        actions.append(experience.action)
        rewards.append(experience.reward)
        dones.append(experience.last_state is None)

        if experience.last_state is None:
            last_state = experience.state
        else:
            last_state = experience.last_state

        last_states.append(last_state)

    return (states, np.array(actions), np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.uint8), last_states)


def obtain_dqn_loss(batch, net: DQNModel, target_net: DQNModel,
                    preprocessor: MAgentPreprocessor, gamma: float,
                    device: Union[torch.device, str] = "cpu"):
    original_states, actions, rewards, dones, original_next_states = unpack_batch(batch)

    states: Tuple[torch.tensor, torch.tensor] = preprocessor(original_states)
    next_states: Tuple[torch.tensor, torch.tensor] = preprocessor(original_next_states)

    actions_tensor: torch.tensor = torch.tensor(actions).to(device)
    rewards_tensor: torch.tensor = torch.tensor(rewards).to(device)
    done_mask: ByteTensor = torch.ByteTensor(dones).to(device)

    actions_tensor: torch.tensor = actions_tensor.unsqueeze(-1)
    state_action_values: torch.tensor = net(states).gather(1, actions_tensor)
    state_action_values = state_action_values.squeeze(-1)

    with torch.no_grad():
        next_state_values = target_net(next_states).max(1)[0]
        next_state_values[done_mask] = 0.0

    bellman_values = next_state_values.detach() * gamma + rewards_tensor
    return nn.MSELoss()(state_action_values, bellman_values)
