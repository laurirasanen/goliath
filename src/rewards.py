from pprint import pprint
from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.reward_functions.combined_reward import CombinedReward
from rlgym_tools.extra_rewards.diff_reward import DiffReward
from rlgym.utils.common_values import (
    BACK_NET_Y,
    BACK_WALL_Y,
    BALL_MAX_SPEED,
    BALL_RADIUS,
    BLUE_GOAL_BACK,
    BLUE_GOAL_CENTER,
    BLUE_TEAM,
    CAR_MAX_SPEED,
    GOAL_HEIGHT,
    ORANGE_GOAL_BACK,
    ORANGE_GOAL_CENTER,
    ORANGE_TEAM,
    SIDE_WALL_X,
)
from rlgym.utils import math
from rlgym.utils.gamestates import PlayerData, GameState
import numpy as np


class GroundedReward(RewardFunction):
    def __init__(self, negative=False):
        super().__init__()
        self.negative = negative

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if player.on_ground:
            return 1.0

        return -1.0 if self.negative else 0.0


class FaceTowardsVelocity(RewardFunction):
    def __init__(self, negative=False):
        super().__init__()
        self.negative = negative

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        val = float(
            np.dot(
                player.car_data.forward(),
                player.car_data.linear_velocity / CAR_MAX_SPEED,
            )
        )
        if not self.negative and val < 0:
            val = 0
        return val


class VelocityReward(RewardFunction):
    # Simple reward function to ensure the model is training.
    def __init__(self, negative=False):
        super().__init__()
        self.negative = negative

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        return (
            np.linalg.norm(player.car_data.linear_velocity)
            / CAR_MAX_SPEED
            * (1 - 2 * self.negative)
        )


class AlignBallGoal(RewardFunction):
    def __init__(self, defense=1.0, offense=1.0):
        super().__init__()
        self.defense = defense
        self.offense = offense

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        ball = state.ball.position
        pos = player.car_data.position
        protecc = np.array(BLUE_GOAL_BACK)
        attacc = np.array(ORANGE_GOAL_BACK)
        if player.team_num == ORANGE_TEAM:
            protecc, attacc = attacc, protecc

        # Align player->ball and net->player vectors
        defensive_reward = self.defense * math.cosine_similarity(
            ball - pos, pos - protecc
        )

        # Align player->ball and player->net vectors
        offensive_reward = self.offense * math.cosine_similarity(
            ball - pos, attacc - pos
        )

        return defensive_reward + offensive_reward


class BoostReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        # 1 reward for each frame with 100 boost, sqrt because 0->20 makes bigger difference than 80->100
        return np.sqrt(player.boost_amount)


class KickoffReward(RewardFunction):
    """
    a simple reward that encourages driving towards the ball fast while it's in the neutral kickoff position
    """

    def __init__(self):
        super().__init__()
        self.vel_dir_reward = VelocityPlayerToBallReward()

    def reset(self, initial_state: GameState):
        self.vel_dir_reward.reset(initial_state)

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = 0
        if state.ball.position[0] == 0 and state.ball.position[1] == 0:
            reward += self.vel_dir_reward.get_reward(player, state, previous_action)
        return reward


class LiuDistancePlayerToBallReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        # Compensate for inside of ball being unreachable (keep max reward at 1)
        dist = (
            np.linalg.norm(player.car_data.position - state.ball.position) - BALL_RADIUS
        )
        return np.exp(
            -0.5 * dist / CAR_MAX_SPEED
        )  # Inspired by https://arxiv.org/abs/2105.12196


class VelocityPlayerToBallReward(RewardFunction):
    def __init__(self, use_scalar_projection=False):
        super().__init__()
        self.use_scalar_projection = use_scalar_projection

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        vel = player.car_data.linear_velocity
        pos_diff = state.ball.position - player.car_data.position
        if self.use_scalar_projection:
            # Vector version of v=d/t <=> t=d/v <=> 1/t=v/d
            # Max value should be max_speed / ball_radius = 2300 / 92.75 = 24.8
            # Used to guide the agent towards the ball
            inv_t = math.scalar_projection(vel, pos_diff)
            return inv_t
        else:
            # Regular component velocity
            norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
            vel /= CAR_MAX_SPEED
            return float(np.dot(norm_pos_diff, vel))


class FaceBallReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        pos_diff = state.ball.position - player.car_data.position
        norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
        return float(np.dot(player.car_data.forward(), norm_pos_diff))


class TouchBallReward(RewardFunction):
    def __init__(self, aerial_weight=0.0):
        self.aerial_weight = aerial_weight

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if player.ball_touched:
            # Default just rewards 1, set aerial weight to reward more depending on ball height
            return (
                (state.ball.position[2] + BALL_RADIUS) / (2 * BALL_RADIUS)
            ) ** self.aerial_weight
        return 0


class BangReward(RewardFunction):
    def __init__(self):
        self.prev_ball_vel = np.zeros(3)

    def reset(self, initial_state: GameState):
        self.prev_ball_vel = np.zeros(3)

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if player.ball_touched:
            vel_diff = np.linalg.norm(state.ball.linear_velocity - self.prev_ball_vel)
            return (
                vel_diff
                * np.sqrt(np.linalg.norm(self.prev_ball_vel))
                / np.sqrt(BALL_MAX_SPEED)
            )  # diff matters more at lower speeds

        self.prev_ball_vel = state.ball.linear_velocity
        return 0


class ShootReward(RewardFunction):
    def __init__(self, power=2.0):
        self.power = power
        self.prev_ball_vel = np.zeros(3)

    def reset(self, initial_state: GameState):
        self.prev_ball_vel = np.zeros(3)

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if player.ball_touched:
            # this target is actually above the goal,
            # but hopefully it encourages higher shots since
            # the ball will drop during flight.
            if player.team_num == 0:
                target_pos = np.array(ORANGE_GOAL_BACK) + [0, 0, GOAL_HEIGHT]
            else:
                target_pos = np.array(BLUE_GOAL_BACK) + [0, 0, GOAL_HEIGHT]

            target_dir = target_pos - state.ball.position
            target_dir /= np.linalg.norm(target_dir)

            prev_dot = np.dot(self.prev_ball_vel, target_dir)
            new_dot = np.dot(state.ball.linear_velocity, target_dir)
            diff = new_dot - prev_dot
            # 1000 unit/s diff -> 1.0 reward
            # 12000 unit/s diff -> 144.0 reward (BALL_MAX_SPEED * 2.0)
            return (diff ** self.power) / (1000 ** self.power)

        self.prev_ball_vel = state.ball.linear_velocity
        return 0


class ClearReward(RewardFunction):
    def __init__(self):
        self.prev_ball_vel = np.zeros(3)

    def reset(self, initial_state: GameState):
        self.prev_ball_vel = np.zeros(3)

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if player.ball_touched:
            clear_radius = SIDE_WALL_X
            if player.team_num == 0:
                goal_pos = BLUE_GOAL_BACK
                clear_direction = [0, 1, 0]
            else:
                goal_pos = ORANGE_GOAL_BACK
                clear_direction = [0, -1, 0]

            # is ball near-ish our goal?
            if np.linalg.norm(goal_pos - state.ball.position) <= clear_radius:
                # compare direction dot to previous.
                # we want to reward clears to the side even if ball is
                # still moving towards our side of the field.
                prev_dot = np.dot(self.prev_ball_vel, clear_direction)
                new_dot = np.dot(state.ball.linear_velocity, clear_direction)
                is_negative = new_dot < prev_dot
                return np.sqrt(abs(new_dot - prev_dot)) * (-1 if is_negative else 1)

        self.prev_ball_vel = state.ball.linear_velocity
        return 0


class LiuDistanceBallToGoalReward(RewardFunction):
    def __init__(self, own_goal=False):
        super().__init__()
        self.own_goal = own_goal

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if (
            player.team_num == BLUE_TEAM
            and not self.own_goal
            or player.team_num == ORANGE_TEAM
            and self.own_goal
        ):
            objective = np.array(ORANGE_GOAL_BACK) + [0, 0, 0.5 * GOAL_HEIGHT]
        else:
            objective = np.array(BLUE_GOAL_BACK) + [0, 0, 0.5 * GOAL_HEIGHT]

        # Compensate for moving objective to back of net
        dist = np.linalg.norm(state.ball.position - objective) - (
            BACK_NET_Y - BACK_WALL_Y + BALL_RADIUS
        )
        return np.exp(
            -0.5 * dist / BALL_MAX_SPEED
        )  # Inspired by https://arxiv.org/abs/2105.12196


class VelocityBallToGoalReward(RewardFunction):
    def __init__(self, own_goal=False, use_scalar_projection=False):
        super().__init__()
        self.own_goal = own_goal
        self.use_scalar_projection = use_scalar_projection

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if (
            player.team_num == BLUE_TEAM
            and not self.own_goal
            or player.team_num == ORANGE_TEAM
            and self.own_goal
        ):
            objective = np.array(ORANGE_GOAL_BACK) + [0, 0, 0.5 * GOAL_HEIGHT]
        else:
            objective = np.array(BLUE_GOAL_BACK) + [0, 0, 0.5 * GOAL_HEIGHT]

        vel = state.ball.linear_velocity
        pos_diff = objective - state.ball.position
        if self.use_scalar_projection:
            # Vector version of v=d/t <=> t=d/v <=> 1/t=v/d
            # Max value should be max_speed / ball_radius = 2300 / 94 = 24.5
            # Used to guide the agent towards the ball
            inv_t = math.scalar_projection(vel, pos_diff)
            return inv_t
        else:
            # Regular component velocity
            norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
            vel /= BALL_MAX_SPEED
            return float(np.dot(norm_pos_diff, vel))


class DoubleEventReward(RewardFunction):
    def __init__(
        self,
        goal=0.0,
        touch=0.0,
        shot=0.0,
        save=0.0,
        demo=0.0,
        enemy_goal=0.0,
        enemy_touch=0.0,
        enemy_shot=0.0,
        enemy_save=0.0,
        enemy_demo=0.0,
    ):
        """
        :param goal: reward for goal scored by player.
        :param touch: reward for touching the ball.
        :param shot: reward for shooting the ball (as detected by Rocket League).
        :param save: reward for saving the ball (as detected by Rocket League).
        :param demo: reward for demolishing a player.
        :param enemy_goal: reward for goal scored by enemy.
        :param enemy_touch: reward for enemy touching the ball.
        :param enemy_shot: reward for enemy shooting the ball (as detected by Rocket League).
        :param enemy_save: reward for enemy saving the ball (as detected by Rocket League).
        :param enemy_demo: reward for enemy demolishing a player.
        """
        super().__init__()
        self.weights = np.array(
            [
                goal,
                touch,
                shot,
                save,
                demo,
                enemy_goal,
                enemy_touch,
                enemy_shot,
                enemy_save,
                enemy_demo,
            ]
        )

        # Need to keep track of last registered value to detect changes
        self.last_registered_values = {}

    @staticmethod
    def _extract_values(player: PlayerData, enemy_player: PlayerData, state: GameState):
        if player.team_num == BLUE_TEAM:
            team_goals, opponent_goals = state.blue_score, state.orange_score
        else:
            team_goals, opponent_goals = state.orange_score, state.blue_score

        enemy_touched = False
        enemy_shots = 0
        enemy_saves = 0
        enemy_demolishes = 0
        if enemy_player is not None:
            enemy_touched = enemy_player.ball_touched
            enemy_shots = enemy_player.match_shots
            enemy_saves = enemy_player.match_saves
            enemy_demolishes = enemy_player.match_demolishes

        return np.array(
            [
                team_goals,
                player.ball_touched,
                player.match_shots,
                player.match_saves,
                player.match_demolishes,
                opponent_goals,
                enemy_touched,
                enemy_shots,
                enemy_saves,
                enemy_demolishes,
            ]
        )

    def reset(self, initial_state: GameState, optional_data=None):
        # Update every reset since rocket league may crash and be restarted with clean values
        self.last_registered_values = {}
        for player in initial_state.players:
            self.last_registered_values[player.car_id] = self._extract_values(
                player, self.get_enemy(player, initial_state), initial_state
            )

    def get_enemy(self, player: PlayerData, state: GameState):
        enemy_player = None
        for p in state.players:
            if p.car_id != player.car_id and p.team_num != player.team_num:
                enemy_player = p
                break
        return enemy_player

    def get_reward(
        self,
        player: PlayerData,
        state: GameState,
        previous_action: np.ndarray,
        optional_data=None,
    ):
        old_values = self.last_registered_values[player.car_id]
        new_values = self._extract_values(player, self.get_enemy(player, state), state)

        diff_values = new_values - old_values
        diff_values[diff_values < 0] = 0  # We only care about increasing values

        reward = np.dot(self.weights, diff_values)

        self.last_registered_values[player.car_id] = new_values
        return reward


class DistanceToFutureBallReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        # TODO: velocity towards ball predict in x time
        return 0


class DribbleReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.prev_touch = 0

    def reset(self, initial_state: GameState):
        self.prev_touch = 0

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        touch_reward = 0.0

        # more frequent touches = good
        if player.ball_touched:
            since_touch = 1.0

            # TODO
            # if state.game_info != None:
            # since_touch = state.game_info.frame_num - self.prev_touch
            # self.prev_touch = state.game_info.frame_num

            touch_reward = 1.0 / max(since_touch, 0.1)

        to_ball = state.ball.position - player.car_data.position
        dist = np.linalg.norm(to_ball)

        # close to ball = good
        near_reward = 4.0 / max(dist, BALL_RADIUS)

        # ball near above = good
        above_dot = np.dot([0, 0, 1], to_ball)
        dist_from_above = abs((2.5 * BALL_RADIUS) - above_dot)
        above_reward = 5.0 / max(dist_from_above, 0.1)

        return touch_reward + near_reward + above_reward


class LandOnWheelsReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if (
            not player.on_ground
            and player.car_data.linear_velocity[2] < 0
            and player.car_data.position[2] < 50.0
        ):
            return np.dot(player.car_data.up(), [0, 0, 1]) * np.dot(
                player.car_data.linear_velocity, [0, 0, -1]
            )

        return 0


accel_reward = DiffReward(VelocityReward(), 0.5)
pickup_boost_reward = DiffReward(BoostReward(), 0.8)


def get_reward_function(scenario: str = "default"):
    funcs = (
        pickup_boost_reward,
        accel_reward,
        VelocityReward(),
        TouchBallReward(aerial_weight=5.0),
        BangReward(),
        ShootReward(power=6.0),
        DribbleReward(),
        ClearReward(),
        AlignBallGoal(),
        DoubleEventReward(
            # us
            1000,
            0,  # separate touch reward
            0,  # separate shot reward
            500,
            200,
            # enemy
            -100,
            -1,
            -10,
            -10,
            -300,  # stop getting demo'd you dummy
        ),
        FaceBallReward(),
        LiuDistanceBallToGoalReward(),
        VelocityBallToGoalReward(),
        VelocityPlayerToBallReward(),
        KickoffReward(),
        GroundedReward(),
        FaceTowardsVelocity(True),
        LiuDistancePlayerToBallReward(),
        DistanceToFutureBallReward(),
        LandOnWheelsReward(),
    )

    # these are messed up
    weights = (
        0.1,  # pickup_boost_reward
        50.0,  # accel_reward
        5000.0,  # VelocityReward
        50.0,  # TouchBallReward
        100.0,  # BangReward
        10000.0,  # ShootReward
        1.0,  # DribbleReward
        1000.0,  # ClearReward
        300.0,  # AlignBallGoal
        1.0,  # DoubleEventReward
        50.0,  # FaceBallReward
        80.0,  # LiuDistanceBallToGoalReward
        175.0,  # VelocityBallToGoalReward
        2000.0,  # VelocityPlayerToBallReward
        100.0,  # KickoffReward
        0.15,  # GroundedReward
        250.0,  # FaceTowardsVelocity
        20.0,  # LiuDistancePlayerToBallReward
        1.0,  # DistanceToFutureBallReward
        20.0,  # LandOnWheelsReward
    )

    return CombinedReward(
        funcs,
        weights,
    )
