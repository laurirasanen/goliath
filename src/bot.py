from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket
from rlgym_compat import GameState

import numpy as np

from agent import Agent
from constants import TICK_SKIP
from obs.goliath_obs import GoliathObs


class GoliathBot(BaseAgent):
    def __init__(self, name, team, index):
        super().__init__(name, team, index)

        self.obs_builder = GoliathObs()
        # Your neural network logic goes inside the Agent class, go take a look inside src/agent.py
        self.agent = Agent()
        # Adjust the tickskip if your agent was trained with a different value
        self.tick_skip = TICK_SKIP

        self.game_state: GameState = None
        self.controls = None
        self.action = None
        self.ticks = 0
        self.update_action = True
        self.prev_time = 0
        print("GoliathBot Ready - Index:", index)

    def initialize_agent(self):
        # Initialize the GameState object now that the game is active and the info is available
        self.game_state = GameState(self.get_field_info())
        self.ticks = self.tick_skip  # So we take an action the first tick
        self.update_action = True
        self.prev_time = 0
        self.controls = SimpleControllerState()
        self.action = np.zeros(8)

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        cur_time = packet.game_info.seconds_elapsed
        delta = cur_time - self.prev_time
        self.prev_time = cur_time

        ticks_elapsed = delta // 0.008  # Smaller than 1/120 on purpose
        self.ticks += ticks_elapsed
        self.game_state.decode(packet, ticks_elapsed)

        if self.update_action:
            self.update_action = False

            # By default we treat every match as a 1v1 against a fixed opponent,
            # by doing this your bot can participate in 2v2 or 3v3 matches. Feel free to change this
            player = self.game_state.players[self.index]
            opponents = [p for p in self.game_state.players if p.team_num != self.team]

            # Here we are are rebuilding the player list as if the match were a 1v1
            self.game_state.players = [player, opponents[0]]

            obs = self.obs_builder.build_obs(player, self.game_state, self.action)
            self.action = self.agent.act(obs, self.game_state)

        if self.ticks >= self.tick_skip:
            self.ticks = 0
            self.update_controls(self.action)
            self.update_action = True

        return self.controls

    def update_controls(self, action):
        self.controls.throttle = action[0]
        self.controls.steer = action[1]
        self.controls.pitch = action[2]
        self.controls.yaw = action[3]
        self.controls.roll = action[4]
        self.controls.jump = action[5] > 0
        self.controls.boost = action[6] > 0
        self.controls.handbrake = action[7] > 0
