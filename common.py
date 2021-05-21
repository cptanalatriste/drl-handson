import warnings
from datetime import timedelta, datetime
from types import SimpleNamespace
from typing import Iterable, List

from ignite.contrib.handlers import TensorboardLogger
from ignite.contrib.handlers.tensorboard_logger import OutputHandler
from ignite.engine import Engine
from ignite.metrics import RunningAverage
from ptan.actions import EpsilonGreedyActionSelector
from ptan.experience import ExperienceSourceFirstLast, ExperienceReplayBuffer
from ptan.ignite import EndOfEpisodeHandler, EpisodeFPSHandler, EpisodeEvents, PeriodicEvents, PeriodEvents


class EpsilonTracker:

    def __init__(self, action_selector: EpsilonGreedyActionSelector,
                 parameters: SimpleNamespace):
        self.action_selector: EpsilonGreedyActionSelector = action_selector
        self.parameters = parameters
        self.frame(0)

    def frame(self, frame_index: int):
        epsilon: float = self.parameters.epsilon_start - frame_index / self.parameters.epsilon_frames
        self.action_selector.epsilon = max(self.parameters.epsilon_final, epsilon)


def setup_ignite(engine: Engine, parameters: SimpleNamespace,
                 experience_source: ExperienceSourceFirstLast,
                 run_name: str, extra_metrics: Iterable[str] = ()):
    warnings.simplefilter("ignore", category=UserWarning)
    handler: EndOfEpisodeHandler = EndOfEpisodeHandler(experience_source, bound_avg_reward=parameters.stop_reward)
    handler.attach(engine)
    EpisodeFPSHandler().attach(engine)

    @engine.on(EpisodeEvents.EPISODE_COMPLETED)
    def episode_completed(trainer: Engine):
        time_passed = trainer.state.metrics.get('time_passed', 0)
        print("Episode %d: reward=%.0f, steps=%s, "
              "speed=%.1f f/s, elapsed=%s" % (
                  trainer.state.episode, trainer.state.episode_reward,
                  trainer.state.episode_steps,
                  trainer.state.metrics.get('avg_fps', 0),
                  timedelta(seconds=int(time_passed))))

    @engine.on(EpisodeEvents.BOUND_REWARD_REACHED)
    def game_solved(trainer: Engine):
        time_passed = trainer.state.metrics['time_passed']
        print("Game solved in %s, after %d episodes "
              "and %d iterations!" % (
                  timedelta(seconds=int(time_passed)),
                  trainer.state.episode, trainer.state.iteration))
        trainer.should_terminate = True

    now = datetime.now().isoformat(timespec='minutes')
    log_directory: str = f"runs/{now}-{parameters.run_name}-{run_name}"
    tensorboard_logger: TensorboardLogger = TensorboardLogger(log_dir=log_directory)
    running_average: RunningAverage = RunningAverage(output_transform=lambda v: v['loss'])
    running_average.attach(engine, "avg_loss")

    metrics: List[str] = ['reward', 'steps', 'avg_reward']
    output_handler: OutputHandler = OutputHandler(tag="episodes", metric_names=metrics)
    episode_completed_event = EpisodeEvents.EPISODE_COMPLETED
    tensorboard_logger.attach(engine, log_handler=output_handler, event_name=episode_completed_event)

    PeriodicEvents().attach(engine)
    metrics = ['avg_loss', 'avg_fps']
    metrics.extend(extra_metrics)
    output_handler = OutputHandler(tag="train", metric_names=metrics, output_transform=lambda a: a)
    hundred_iterations_event = PeriodEvents.ITERS_100_COMPLETED
    tensorboard_logger.attach(engine, log_handler=output_handler, event_name=hundred_iterations_event)


def batch_generator(replay_buffer: ExperienceReplayBuffer,
                    initial: int, batch_size: int):
    replay_buffer.populate(initial)
    while True:
        replay_buffer.populate(1)
        yield replay_buffer.sample(batch_size)
