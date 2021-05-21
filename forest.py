# Workaround for Google Collab
import sys
sys.path.insert(1, "/content/gdrive/My Drive/google_colab/MAgent/python")

import os
from types import SimpleNamespace
from typing import Dict, List, Tuple, Union

import ptan.agent
import torch
from ignite.engine import Engine
from magent import GridWorld
from ptan.actions import EpsilonGreedyActionSelector, ArgmaxActionSelector
from ptan.agent import TargetNet
from ptan.experience import ExperienceFirstLast
from ptan.ignite import PeriodEvents
from torch import optim

from common import setup_ignite, batch_generator
from common import EpsilonTracker
from data import MAgentEnv
from model import DQNModel, MAgentPreprocessor, obtain_dqn_loss

MAP_SIZE = 64
COUNT_DEERS = 50
COUNT_TIGERS = 10
WALLS_DENSITY = 0.04

PARAMETERS = SimpleNamespace(**{
    'epsilon_start': 1.0,
    'epsilon_final': 0.02,
    'epsilon_frames': 5 * 10 ** 5,
    'gamma': 0.99,
    'replay_size': 1000000,
    'learning_rate': 1e-4,
    'target_net_sync': 1000,
    'stop_reward': None,
    'run_name': 'tigers',
    'replay_initial': 100,
    'batch_size': 32

})


def test_model(dqn_model: DQNModel, device: torch.device, mode: str) -> Tuple[float, float]:
    gridworld_test: GridWorld = GridWorld(mode, map_size=MAP_SIZE)
    deer_handle: int
    tiger_handle: int
    deer_handle, tiger_handle = gridworld_test.get_handles()

    def reset_environment():
        gridworld_test.reset()
        gridworld_test.add_walls(method="random", n=MAP_SIZE * MAP_SIZE * WALLS_DENSITY)
        gridworld_test.add_agents(deer_handle, method="random", n=COUNT_DEERS)
        gridworld_test.add_agents(tiger_handle, method="random", n=COUNT_TIGERS)

    magent_environment: MAgentEnv = MAgentEnv(gridworld_test, tiger_handle,
                                              reset_environment_funcion=reset_environment)
    pre_processor: MAgentPreprocessor = MAgentPreprocessor(device)
    dqn_agent: ptan.agent.DQNAgent = ptan.agent.DQNAgent(dqn_model, ArgmaxActionSelector(),
                                                         device, preprocessor=pre_processor)

    observation = magent_environment.reset()
    steps: int = 0
    rewards: float = 0.0

    while True:
        actions = dqn_agent(observation)[0]
        observations, all_rewards, dones, _ = magent_environment.step(actions)
        steps += len(observations)
        rewards += sum(all_rewards)
        if dones[0]:
            break

    return rewards / COUNT_TIGERS, steps / COUNT_TIGERS


if __name__ == "__main__":
    cuda: bool = True  # Modify as required
    run_name: str = "run"
    mode: str = "forest"

    device: torch.device = torch.device("cuda" if cuda else "cpu")
    saves_path = os.path.join("saves", run_name)
    os.makedirs(saves_path, exist_ok=True)

    gridworld: GridWorld = GridWorld(mode, map_size=MAP_SIZE)

    deer_handle: int
    tiger_handle: int
    deer_handle, tiger_handle = gridworld.get_handles()


    def reset_environment():
        gridworld.reset()
        gridworld.add_walls(method="random", n=MAP_SIZE * MAP_SIZE * WALLS_DENSITY)
        gridworld.add_agents(deer_handle, method="random", n=COUNT_DEERS)
        gridworld.add_agents(tiger_handle, method="random", n=COUNT_TIGERS)


    environment: MAgentEnv = MAgentEnv(gridworld, tiger_handle, reset_environment_funcion=reset_environment)

    dqn_model: DQNModel = DQNModel(environment.single_observation_space.spaces[0].shape,
                                   environment.single_observation_space.spaces[1].shape,
                                   gridworld.get_action_space(tiger_handle)[0]).to(device)

    target_net: TargetNet = ptan.agent.TargetNet(dqn_model)
    print(target_net)

    action_selector: EpsilonGreedyActionSelector = EpsilonGreedyActionSelector(epsilon=PARAMETERS.epsilon_start)
    epsilon_tracker: EpsilonTracker = EpsilonTracker(action_selector, PARAMETERS)
    pre_processor: MAgentPreprocessor = MAgentPreprocessor(device)
    dqn_agent: ptan.agent.DQNAgent = ptan.agent.DQNAgent(dqn_model, action_selector, device,
                                                         preprocessor=pre_processor)
    experience_source: ptan.experience.ExperienceSourceFirstLast = ptan.experience.ExperienceSourceFirstLast(
        environment, dqn_agent, PARAMETERS.gamma, vectorized=True)
    replay_buffer: ptan.experience.ExperienceReplayBuffer = ptan.experience.ExperienceReplayBuffer(
        experience_source, PARAMETERS.replay_size)
    optimizer: optim.Adam = optim.Adam(dqn_model.parameters(), lr=PARAMETERS.learning_rate)


    def process_batch(engine: Engine, batch: List[ExperienceFirstLast]):
        result: Dict = {}
        optimizer.zero_grad()
        loss = obtain_dqn_loss(batch, dqn_model, target_net.target_model, pre_processor,
                               gamma=PARAMETERS.gamma, device=device)
        loss.backward()
        optimizer.step()

        epsilon_tracker.frame(engine.state.iteration)
        result['epsilon'] = action_selector.epsilon

        if engine.state.iteration % PARAMETERS.target_net_sync == 0:
            target_net.sync()

        result['loss'] = loss.item()
        return result


    engine = Engine(process_batch)
    setup_ignite(engine, PARAMETERS, experience_source, run_name, extra_metrics=('test_reward', 'test_steps'))

    best_test_reward: Union[float, None]
    best_test_reward = None


    @engine.on(PeriodEvents.ITERS_10000_COMPLETED)
    def test_network(engine: Engine):
        dqn_model.train(False)
        test_reward: float
        test_steps: float
        test_reward, test_steps = test_model(dqn_model, device, mode)
        dqn_model.train(True)

        engine.state.metrics['test_reward'] = test_reward
        engine.state.metrics['test_steps'] = test_steps
        print("Test done: got %.3f reward after %.2f steps" % (test_reward, test_steps))

        global best_test_reward
        if best_test_reward is None:
            best_test_reward = test_reward
        elif best_test_reward < test_reward:
            print("Best test reward updated %.3f <- %.3f, save model" % (best_test_reward, test_reward))
            best_test_reward = test_reward
            torch.save(dqn_model.state_dict(), os.path.join(saves_path, "best_%.3f.dat" % test_reward))


    engine.run(batch_generator(replay_buffer, PARAMETERS.replay_initial, PARAMETERS.batch_size))
