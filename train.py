import datetime
import time
from argparse import ArgumentParser
import numpy as np

# Here we import the Match object and our multi-instance wrapper
from rlgym.envs import Match
from rlgym.utils.action_parsers.continuous_act import ContinuousAction
from rlgym.utils.action_parsers.discrete_act import DiscreteAction
from rlgym.utils.common_values import BALL_RADIUS
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv

# Since we can't use the normal rlgym.make() function, we need to import all the default configuration objects to give to our Match.
from rlgym.utils.terminal_conditions.common_conditions import (
    GoalScoredCondition,
    NoTouchTimeoutCondition,
    TimeoutCondition,
)

# from rlgym_tools.sb3_utils.sb3_multidiscrete_wrapper import SB3MultiDiscreteWrapper
from stable_baselines3.common.vec_env.vec_check_nan import VecCheckNan
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

# Finally, we import the SB3 implementation of PPO.
from stable_baselines3.ppo import PPO

from src.conditions import NoTouchOrRangeTimeoutCondition
from src.constants import (
    HALF_LIFE_SECONDS,
    PHYSICS_TICKRATE,
    TICK_SKIP,
    TRAIN_DEVICE,
    seconds_to_steps,
)
from src.obs.goliath_obs import GoliathObs
from src.rewards import get_reward_function
from src.state_setters import DribbleState, KickoffState, AttackState, ShootState
from src.util import get_latest_path


def get_matches(args):
    matches = []
    scenario = args.scenario

    for x in range(args.environments):
        # Handle switching scenarios if "mixed"
        if args.scenario == "mixed":
            # 1 kickoff instances
            if x == 0:
                scenario = "kickoff"
            # 2 default
            elif x == 1:
                scenario = "default"
            # 1 random attack
            elif x == 3:
                scenario = "attack"
            # 1 dribble
            elif x == 4:
                scenario = "dribble"
            # rest shooting practice
            elif x == 5:
                scenario = "shoot"

        # defaults
        state_setter = KickoffState()
        terminal_conditions = [
            NoTouchTimeoutCondition(seconds_to_steps(8)),
            GoalScoredCondition(),
        ]
        reward_function = get_reward_function(scenario)

        # scenario specifics
        if scenario == "default":
            pass
        elif scenario == "attack":
            state_setter = AttackState()
        elif scenario == "kickoff":
            terminal_conditions = [
                TimeoutCondition(seconds_to_steps(4)),
            ]
        elif scenario == "dribble":
            state_setter = DribbleState()
            terminal_conditions = [
                NoTouchOrRangeTimeoutCondition(seconds_to_steps(2), 3.0 * BALL_RADIUS),
                GoalScoredCondition(),
            ]
        elif scenario == "shoot":
            state_setter = ShootState()
            terminal_conditions = [
                NoTouchTimeoutCondition(seconds_to_steps(4)),
                GoalScoredCondition(),
            ]
        else:
            raise f"Unknown scenario {scenario}"

        # create match
        matches.append(
            Match(
                reward_function=reward_function,
                terminal_conditions=terminal_conditions,
                obs_builder=GoliathObs(),
                action_parser=DiscreteAction(),
                state_setter=state_setter,
                self_play=args.self_play,
                spawn_opponents=not args.self_play,
                tick_skip=TICK_SKIP,
                game_speed=100,
            )
        )

    return matches


# This is the function we need to provide to our SB3MultipleInstanceEnv to construct a match. Note that this function MUST return a Match object.
def get_model(args):
    fps = PHYSICS_TICKRATE / TICK_SKIP
    gamma = np.exp(np.log(0.5) / (fps * HALF_LIFE_SECONDS))  # Quick mafs
    print(f"fps={fps}, gamma={gamma})")

    # Here we configure our Match. If you want to use custom configuration objects, make sure to replace the default arguments here with instances of the objects you want.
    env = SB3MultipleInstanceEnv(
        match_func_or_matches=get_matches(args),
        num_instances=args.environments,
        wait_time=args.wait,
    )
    # env = SB3MultiDiscreteWrapper(env)
    env = VecCheckNan(env)  # Optional
    env = VecMonitor(env)  # Recommended, logs mean reward and ep_len to Tensorboard
    env = VecNormalize(
        env, norm_obs=False, gamma=gamma
    )  # Highly recommended, normalizes rewards

    num_timesteps = 0
    elapsed_time = 0
    filepath = get_latest_path()
    if filepath is not None:
        model = PPO.load(
            path=filepath,
            env=env,
            custom_objects=dict(n_envs=env.num_envs),  # needed for changing num envs
            device=TRAIN_DEVICE,
        )
        print(f"Loaded model {filepath}")
        # Assume steps and elapsed time in filename
        steps_str = filepath.split("_")[-2]
        time_str = filepath.split("_")[-1][:-4]
        print(f"Steps: {steps_str}, time: {time_str}")
        num_timesteps = int(steps_str)
        elapsed_time = int(time_str)
    else:
        print("Creating new model")
        # Hyperparameters presumably better than default; inspired by original PPO paper
        model = PPO(
            policy="MlpPolicy",
            env=env,
            n_epochs=32,  # PPO calls for multiple epochs
            learning_rate=1e-5,  # Around this is fairly common for PPO
            ent_coef=0.01,  # From PPO Atari
            vf_coef=1.0,  # From PPO Atari
            gamma=gamma,  # Gamma as calculated using half-life
            verbose=3,  # Print out all the info as we're going
            batch_size=4096,  # Batch size as high as possible within reason
            n_steps=4096,  # Number of steps to perform before optimizing network
            tensorboard_log="out/logs",  # `tensorboard --logdir out/logs` in terminal to see graphs
            device=TRAIN_DEVICE,
        )

    return model, num_timesteps, elapsed_time


# If we want to spawn new processes, we have to make sure our program starts in a proper Python entry point.
if __name__ == "__main__":
    """
    Now all we have to do is make an instance of the SB3MultipleInstanceEnv and pass it our get_match function, the number of instances we'd like to open, and how long it should wait between instances.
    This wait_time argument is important because if multiple Rocket League clients are opened in quick succession, they will cause each other to crash. The exact reason this happens is unknown to us,
    but the easiest solution is to delay for some period of time between launching clients. The amount of required delay will depend on your hardware, so make sure to change this number if your Rocket League
    clients are crashing before they fully launch.
    """
    parser = ArgumentParser()
    parser.add_argument(
        "-s",
        "--scenario",
        dest="scenario",
        help="Scenario to train for",
        type=str,
        default="default",
    )
    parser.add_argument(
        "-e",
        "--environments",
        dest="environments",
        help="Number of environments to launch",
        type=int,
        default=7,
    )
    parser.add_argument(
        "-w",
        "--wait",
        dest="wait",
        help="Wait time between environment launches",
        type=int,
        default=15,
    )
    parser.add_argument(
        "-sp",
        "--self_play",
        dest="self_play",
        help="Play against self, spawn Psyonix bot if false",
        type=bool,
        default=True,
    )
    args = parser.parse_args()
    print(
        f"Training for {args.scenario} scenario with {args.environments} environments"
    )

    model, num_timesteps, elapsed_time = get_model(args)
    SAVE_INTERVAL = round(10_000_000 / model.n_envs)
    while True:
        model.learn(total_timesteps=SAVE_INTERVAL)
        num_timesteps += model.num_timesteps * model.n_envs
        elapsed_time += int(time.time() - model.start_time) * model.n_envs
        filename = f"models/model_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_{num_timesteps}_{elapsed_time}.zip"
        print(f"Saving model to {filename}")
        model.save(filename)
