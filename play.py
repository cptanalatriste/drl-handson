from typing import Tuple

import numpy as np
import torch
from magent.gridworld import Config, GridWorld
from numpy import ndarray

import data
from model import DQNModel

if __name__ == "__main__":
    map_size: int = 64
    render_directory: str = "render"
    wall_density: float = 0.04
    deers: int = 50
    tigers: int = 10
    model: str = "saves/run/best_67.500.dat"
    map_location: torch.device = torch.device("cpu")

    configuration: Config = data.get_forest_configuration(map_size)

    environment: GridWorld = GridWorld(configuration, map_size=map_size)
    environment.set_render_dir(render_directory)

    deer_handle: int
    tiger_handle: int

    deer_handle, tiger_handle = environment.get_handles()

    environment.reset()
    environment.add_walls(method="random", n=map_size * map_size * wall_density)
    environment.add_agents(deer_handle, method="random", n=deers)
    environment.add_agents(tiger_handle, method="random", n=tigers)

    view_space: Tuple = environment.get_view_space(tiger_handle)
    view_space = (view_space[-1],) + view_space[:2]
    dqn_model: DQNModel = DQNModel(view_space, environment.get_feature_space(tiger_handle),
                                   environment.get_action_space(tiger_handle)[0])
    dqn_model.load_state_dict(torch.load(model, map_location=map_location))
    print(dqn_model)

    total_reward: float = 0.0
    while True:
        view_observation, feature_observation = environment.get_observation(tiger_handle)

        view_as_array: ndarray = np.array(view_observation)
        features_as_array: ndarray = np.array(feature_observation)
        view_as_array = np.moveaxis(view_as_array, 3, 1)
        view_as_tensor: torch.tensor = torch.tensor(view_as_array, dtype=torch.float32)
        features_as_tensor: torch.tensor = torch.tensor(features_as_array, dtype=torch.float32)

        q_values = dqn_model((view_as_tensor, features_as_tensor))
        actions: ndarray = torch.max(q_values, dim=1)[1].cpu().numpy()
        actions = actions.astype(np.int32)

        environment.set_action(tiger_handle, actions)
        done: bool = environment.step()
        if done:
            break

        environment.render()
        environment.clear_dead()
        total_reward += environment.get_reward(tiger_handle).sum()

    print(f"Average reward: {total_reward / tigers}")
