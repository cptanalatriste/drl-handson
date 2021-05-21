from typing import List, Tuple

import magent

from magent.builtin.rule_model import RandomActor
from magent.model import BaseModel
from numpy import ndarray

MAP_SIZE: int = 64

if __name__ == "__main__":
    environment: magent.GridWorld = magent.GridWorld("forest", map_size=MAP_SIZE)
    environment.set_render_dir("render")

    deer_handle: int
    tiger_handle: int
    deer_handle, tiger_handle = environment.get_handles()

    models: List[BaseModel] = [RandomActor(environment, deer_handle),
                               RandomActor(environment, tiger_handle)]

    environment.reset()
    environment.add_walls(method="random", n=MAP_SIZE * MAP_SIZE * 0.04)
    environment.add_agents(deer_handle, method="random", n=5)
    environment.add_agents(tiger_handle, method="random", n=2)

    tiger_view_space: Tuple = environment.get_view_space(tiger_handle)
    tiger_feature_space: Tuple = environment.get_feature_space(tiger_handle)
    print(f"Tiger view space: {tiger_view_space} features: {tiger_feature_space}")

    deer_view_space: Tuple = environment.get_view_space(deer_handle)
    deer_feature_space: Tuple = environment.get_feature_space(deer_handle)
    print(f"Deer view space: {deer_view_space} features: {deer_feature_space}")

    done: bool = False
    step: int = 0

    while not done:
        deer_observation: Tuple = environment.get_observation(deer_handle)
        tiger_observation: Tuple = environment.get_observation(tiger_handle)

        if step == 0:
            print(f"Tiger observation {tiger_observation[0].shape} {tiger_observation[1].shape}")
            print(f"Deer observation {deer_observation[0].shape} {deer_observation[1].shape}")

        print(f"{step}: HP deers: {deer_observation[0][:, 1, 1, 2]}")
        print(f"{step} HP tigers: {tiger_observation[0][:, 4, 4, 2]}")

        deer_actions: ndarray = models[0].infer_action(deer_observation)
        tiger_actions: ndarray = models[1].infer_action(tiger_observation)

        environment.set_action(deer_handle, deer_actions)
        environment.set_action(tiger_handle, tiger_actions)

        environment.render()
        done = environment.step()
        environment.clear_dead()
        tiger_reward: ndarray = environment.get_reward(tiger_handle)
        deer_reward: ndarray = environment.get_reward(deer_handle)

        print(f"Rewards: deer {tiger_reward}, tiger {deer_reward}")

        step += 1
