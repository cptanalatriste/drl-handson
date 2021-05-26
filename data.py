from typing import Callable, Optional, Tuple, List, Dict

import gym
import numpy as np
from gym import spaces
from gym.vector import VectorEnv
from magent import GridWorld
from magent.gridworld import Config, CircleRange
from numpy import ndarray


class MAgentEnv(VectorEnv):

    def __init__(self, environment: GridWorld, handle: int,
                 reset_environment_funcion: Callable[[], None],
                 is_slave: bool = False,
                 step_limit: Optional[int] = None):
        self._steps_done: int = 0
        reset_environment_funcion()
        action_space: gym.Space = self.handle_action_space(environment, handle)
        observation_space: gym.Space = self.handle_observation_space(environment, handle)
        number_of_agents: int = environment.get_num(handle)

        super(MAgentEnv, self).__init__(number_of_agents, observation_space,
                                        action_space)

        self.action_space = self.single_action_space
        self._env: GridWorld = environment
        self._handle: int = handle
        self._reset_env_func: Callable[[], None] = reset_environment_funcion
        self._is_slave: bool = is_slave
        self._steps_limit: Optional[int] = step_limit

    @classmethod
    def handle_action_space(cls, environment: GridWorld, handle: int) -> gym.Space:
        return spaces.Discrete(environment.get_action_space(handle)[0])

    @classmethod
    def handle_observation_space(cls, environment: GridWorld, handle: int) -> gym.Space:
        magent_view_space: Tuple = environment.get_view_space(handle)
        magent_feature_space: Tuple = environment.get_feature_space(handle)

        view_shape: Tuple = (magent_view_space[-1],) + magent_view_space[:2]
        view_space: spaces.Box = spaces.Box(low=0.0, high=1.0, shape=view_shape)
        extra_space: spaces.Box = spaces.Box(low=0.0, high=1.0, shape=magent_feature_space)

        return spaces.Tuple((view_space, extra_space))

    @classmethod
    def handle_observations(cls, environment: GridWorld, handle: int) -> List[Tuple[ndarray, ndarray]]:
        view_observation, feature_observation = environment.get_observation(handle)
        entries: int = view_observation.shape[0]
        if entries == 0:
            return []

        view_observation_array: ndarray = np.array(view_observation)
        feature_observation_array: ndarray = np.array(feature_observation)
        view_observation_array = np.moveaxis(view_observation_array, 3, 1)

        result: List[Tuple[ndarray, ndarray]] = []
        for observation, features in zip(np.vsplit(view_observation_array, entries),
                                         np.vsplit(feature_observation_array, entries)):
            result.append((observation[0], features[0]))

        return result

    def reset_wait(self) -> List[Tuple[ndarray, ndarray]]:
        self._steps_done = 0
        if not self._is_slave:
            self._reset_env_func()

        return self.handle_observations(self._env, self._handle)

    def step_async(self, actions) -> None:
        action_array: ndarray = np.array(actions, dtype=np.int32)
        self._env.set_action(self._handle, action_array)

    def step_wait(self) -> Tuple[List[Tuple], List[float], List[bool], Dict]:
        self._steps_done += 1
        if not self._is_slave:
            done: bool = self._env.step()
            self._env.clear_dead()

            if self._steps_limit is not None and self._steps_limit <= self._steps_done:
                done = True
        else:
            done = False

        observations: List[Tuple[ndarray, ndarray]] = self.handle_observations(self._env, self._handle)
        rewards: List[float] = self._env.get_reward(self._handle).tolist()
        dones: List[bool] = [done] * len(rewards)

        if done:
            observations = self.reset()
            dones = [done] * self.num_envs
            rewards = [0.0] * self.num_envs

        return observations, rewards, dones, {}


def get_default_forest_configuration(map_size: int) -> Config:
    configuration: Config = Config()

    configuration.set({"map_width": map_size,
                       "map_height": map_size})
    configuration.set({"embedding_size": 10})

    deer_agent_type: str = configuration.register_agent_type("deer", {'width': 1, 'length': 1, 'hp': 5, 'speed': 1,
                                                                      'view_range': CircleRange(1),
                                                                      'attack_range': CircleRange(0),
                                                                      'damage': 0, 'step_recover': 0.2,
                                                                      'food_supply': 0, 'kill_supply': 8,
                                                                      # added to standard 'forest' setup to motivate deers to live longer :)
                                                                      'step_reward': 1})
    tiger_agent_type: str = configuration.register_agent_type("tiger", {'width': 1, 'length': 1, 'hp': 10, 'speed': 1,
                                                                        'view_range': CircleRange(4),
                                                                        'attack_range': CircleRange(1),
                                                                        'damage': 3, 'step_recover': -0.5,
                                                                        'food_supply': 0, 'kill_supply': 0,
                                                                        'step_reward': 1, 'attack_penalty': -0.1})

    configuration.add_group(deer_agent_type)
    configuration.add_group(tiger_agent_type)

    return configuration
