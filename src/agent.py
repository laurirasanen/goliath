from rlgym_compat import GameState
from rlgym.utils.action_parsers.discrete_act import DiscreteAction
from stable_baselines3.ppo import PPO
from constants import TRAIN_DEVICE

from util import get_latest_path


class Agent:
    def __init__(self):
        self.actor = PPO.load(
            path=get_latest_path(),
            device=TRAIN_DEVICE,
        )
        self.prev_states = None
        self.action_parser = DiscreteAction()

    def act(self, obs: list, game_state: GameState):
        # Evaluate your model here
        action, self.prev_states = self.actor.predict(obs, self.prev_states)
        action = self.action_parser.parse_actions(action, game_state)[0]
        return action
