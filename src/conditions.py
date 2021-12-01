from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition
from rlgym.utils.gamestates import GameState

import numpy as np


class NoTouchOrRangeTimeoutCondition(TimeoutCondition):
    """
    A condition that will terminate an episode after some number of steps,
    unless a player is near the ball.

    NoTouchTimeoutCondition with added check for ball distance.
    Touch event seems to not get invoked for soft continous touches,
    e.g. when dribbling the ball.
    """

    def __init__(self, max_steps: int, min_distance: float):
        super().__init__(max_steps)
        self.min_distance = min_distance

    def is_terminal(self, current_state: GameState):
        if any(p.ball_touched for p in current_state.players):
            self.steps = 0
            return False
        else:
            for p in current_state.players:
                if (
                    np.linalg.norm(current_state.ball.position - p.car_data.position)
                    < self.min_distance
                ):
                    self.steps = 0
                    return False

            return super(NoTouchOrRangeTimeoutCondition, self).is_terminal(
                current_state
            )
