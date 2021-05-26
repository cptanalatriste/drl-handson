from typing import Tuple

import numpy as np
import torch
from magent.gridworld import Config, GridWorld
from numpy import ndarray

import social
from forest import COUNT_DEERS, COUNT_TIGERS, MAP_SIZE
from model import DQNModel


def get_actions(gridworld: GridWorld, tiger_model: DQNModel, group_handle: int) -> ndarray:
    view_observation, feature_observation = gridworld.get_observation(group_handle)

    view_as_array: ndarray = np.array(view_observation)
    features_as_array: ndarray = np.array(feature_observation)
    view_as_array = np.moveaxis(view_as_array, 3, 1)
    view_as_tensor: torch.tensor = torch.tensor(view_as_array, dtype=torch.float32)
    features_as_tensor: torch.tensor = torch.tensor(features_as_array, dtype=torch.float32)

    q_values = tiger_model((view_as_tensor, features_as_tensor))
    actions: ndarray = torch.max(q_values, dim=1)[1].cpu().numpy()
    actions = actions.astype(np.int32)

    return actions


if __name__ == "__main__":
    map_size: int = MAP_SIZE
    deers: int = COUNT_DEERS
    tigers: int = COUNT_TIGERS

    render_directory: str = "render"
    wall_density: float = 0.04

    model: str = "saves/edible_tigers/best_46.600.dat"
    map_location: torch.device = torch.device("cpu")

    configuration: Config = social.get_forest_configuration(map_size)

    environment: GridWorld = GridWorld(configuration, map_size=map_size)
    environment.set_render_dir(render_directory)

    deer_handle: int
    first_tiger_handle: int
    second_tiger_handle: int

    deer_handle, first_tiger_handle, second_tiger_handle = environment.get_handles()

    environment.reset()
    environment.add_walls(method="random", n=map_size * map_size * wall_density)
    environment.add_agents(deer_handle, method="random", n=deers)
    environment.add_agents(first_tiger_handle, method="random", n=tigers)
    environment.add_agents(second_tiger_handle, method="random", n=tigers)

    view_space: Tuple = environment.get_view_space(first_tiger_handle)
    view_space = (view_space[-1],) + view_space[:2]
    dqn_model: DQNModel = DQNModel(view_space, environment.get_feature_space(first_tiger_handle),
                                   environment.get_action_space(first_tiger_handle)[0])
    dqn_model.load_state_dict(torch.load(model, map_location=map_location))
    print(dqn_model)

    reward_tiger_1: float = 0.0
    reward_tiger_2: float = 0.0

    survivors: int
    while True:
        first_tiger_actions: ndarray = get_actions(environment, dqn_model, first_tiger_handle)
        second_tiger_actions: ndarray = get_actions(environment, dqn_model, second_tiger_handle)

        environment.set_action(first_tiger_handle, first_tiger_actions)
        environment.set_action(second_tiger_handle, second_tiger_actions)

        done: bool = environment.step()
        survivors = np.count_nonzero(environment.get_alive(deer_handle))
        if done:
            break

        environment.render()
        environment.clear_dead()
        reward_tiger_1 += environment.get_reward(first_tiger_handle).sum()
        reward_tiger_2 += environment.get_reward(second_tiger_handle).sum()

    print(
        f"Average reward Team 1: {reward_tiger_1 / tigers} Average reward Team 2: {reward_tiger_2 / tigers} Surviving deers {survivors} of {deers}")
