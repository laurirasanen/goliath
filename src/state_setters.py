import random
import numpy as np

from rlgym.utils.state_setters import StateSetter
from rlgym.utils.state_setters import StateWrapper
from rlgym.utils.common_values import (
    BACK_WALL_Y,
    BALL_RADIUS,
    CEILING_Z,
    SIDE_WALL_X,
    CAR_MAX_SPEED,
    CAR_MAX_ANG_VEL,
)


# Fix spawning inside the rounded corners of the arena.
# Prevent x and y both being less than 1024 units away from wall
# at the same time.
# TODO: use the actual curve instead of cutting a rect out of corners
def fix_spawn_position(x: float, y: float):
    min_x = -SIDE_WALL_X + 1024
    max_x = SIDE_WALL_X - 1024
    min_y = -BACK_WALL_Y + 1024
    max_y = BACK_WALL_Y - 1024

    if x < min_x:
        if y < min_y:
            x_diff = x - min_x
            y_diff = y - min_y
            if x_diff < y_diff:
                x = min_x
            else:
                y = min_y
            return x, y

        if y > max_y:
            x_diff = x - min_x
            y_diff = max_y - y
            if x_diff < y_diff:
                x = min_x
            else:
                y = max_y
            return x, y

    if x > max_x:
        if y < min_y:
            x_diff = max_x - x
            y_diff = y - min_y
            if x_diff < y_diff:
                x = max_x
            else:
                y = min_y
            return x, y

        if y > max_y:
            x_diff = max_x - x
            y_diff = max_y - y
            if x_diff < y_diff:
                x = max_x
            else:
                y = max_y
            return x, y

    return x, y


class KickoffState(StateSetter):
    SPAWN_BLUE_POS = [
        [-2048, -2560, 17],
        [2048, -2560, 17],
        [-256, -3840, 17],
        [256, -3840, 17],
        [0, -4608, 17],
    ]
    SPAWN_BLUE_YAW = [0.25 * np.pi, 0.75 * np.pi, 0.5 * np.pi, 0.5 * np.pi, 0.5 * np.pi]
    SPAWN_ORANGE_POS = [
        [2048, 2560, 17],
        [-2048, 2560, 17],
        [256, 3840, 17],
        [-256, 3840, 17],
        [0, 4608, 17],
    ]
    SPAWN_ORANGE_YAW = [
        -0.75 * np.pi,
        -0.25 * np.pi,
        -0.5 * np.pi,
        -0.5 * np.pi,
        -0.5 * np.pi,
    ]

    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        """
        Modifies state_wrapper values to emulate a randomly selected default kickoff.

        :param state_wrapper: StateWrapper object to be modified with desired state values.
        """
        # possible kickoff indices are shuffled
        # 0 = diag right
        # 1 = diag left
        # 2 = center right
        # 3 = center left
        # 4 = center
        spawn_inds = [0, 1, 2, 3, 4]
        # the bot sucks at starting on the right side,
        # force it to train these positions more often.
        # spawn_inds = [0, 0, 0, 0, 1, 2, 2, 3, 4]
        random.shuffle(spawn_inds)

        blue_count = 0
        orange_count = 0
        for car in state_wrapper.cars:
            pos = [0.0, 0.0, 0.0]
            yaw = 0.0

            # team_num = 0 = blue team
            if car.team_num == 0:
                # select a unique spawn state from pre-determined values
                pos = self.SPAWN_BLUE_POS[spawn_inds[blue_count]]
                yaw = self.SPAWN_BLUE_YAW[spawn_inds[blue_count]]
                blue_count += 1

            # team_num = 1 = orange team
            elif car.team_num == 1:
                # select a unique spawn state from pre-determined values
                pos = self.SPAWN_ORANGE_POS[spawn_inds[orange_count]]
                yaw = self.SPAWN_ORANGE_YAW[spawn_inds[orange_count]]
                orange_count += 1

            # set car state values
            pos[0], pos[1] = fix_spawn_position(pos[0], pos[1])
            car.set_pos(*pos)
            car.set_rot(yaw=yaw)
            car.boost = 0.33


class AttackState(StateSetter):
    FLOOR_SPAWN_CHANCE = 0.5

    # Spawn blue near center field
    SPAWN_BLUE_POS_X = [-SIDE_WALL_X + 256, SIDE_WALL_X - 256]
    SPAWN_BLUE_POS_Y = [-1500, 500]
    SPAWN_BLUE_POS_Z = [17, CEILING_Z - 150]
    SPAWN_BLUE_YAW = [0.25 * np.pi, 0.75 * np.pi]
    SPAWN_BLUE_PITCH = [-0.25 * np.pi, 0.25 * np.pi]
    SPAWN_BLUE_ROLL = [-0.25 * np.pi, 0.25 * np.pi]
    SPAWN_BLUE_VELOCITY_X = [-200, 200]
    SPAWN_BLUE_VELOCITY_Y = [-100, 800]
    SPAWN_BLUE_VELOCITY_Z = [-200, 500]
    SPAWN_BLUE_ANG_VELOCITY = [-2.0, 2.0]

    # Spawn orange near goal
    SPAWN_ORANGE_POS_X = [-SIDE_WALL_X + 256, SIDE_WALL_X - 256]
    SPAWN_ORANGE_POS_Y = [BACK_WALL_Y - 512, BACK_WALL_Y - 256]
    SPAWN_ORANGE_POS_Z = [17, CEILING_Z - 150]
    SPAWN_ORANGE_YAW = [-0.25 * np.pi, -0.75 * np.pi]
    SPAWN_ORANGE_PITCH = [-0.25 * np.pi, 0.25 * np.pi]
    SPAWN_ORANGE_ROLL = [-0.25 * np.pi, 0.25 * np.pi]
    SPAWN_ORANGE_VELOCITY_X = [-200, 200]
    SPAWN_ORANGE_VELOCITY_Y = [-800, 100]
    SPAWN_ORANGE_VELOCITY_Z = [-200, 500]
    SPAWN_ORANGE_ANG_VELOCITY = [-2.0, 2.0]

    # Spawn goal somewhere in the middle
    SPAWN_BALL_POS_X = [-SIDE_WALL_X + 256, SIDE_WALL_X - 256]
    SPAWN_BALL_POS_Y = [1000, 1500]
    SPAWN_BALL_POS_Z = [17, CEILING_Z - 128]
    SPAWN_BALL_VELOCITY = [-800, 800]

    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        """
        Modifies state_wrapper values to emulate a randomly selected attack scenario.

        :param state_wrapper: StateWrapper object to be modified with desired state values.
        """

        floor_spawn = random.random() < self.FLOOR_SPAWN_CHANCE

        # set ball pos
        state_wrapper.ball.position[0] = random.uniform(
            self.SPAWN_BALL_POS_X[0], self.SPAWN_BALL_POS_X[1]
        )
        state_wrapper.ball.position[1] = random.uniform(
            self.SPAWN_BALL_POS_Y[0], self.SPAWN_BALL_POS_Y[1]
        )
        if floor_spawn:
            state_wrapper.ball.position[2] = 17 + BALL_RADIUS
        else:
            state_wrapper.ball.position[2] = random.uniform(
                self.SPAWN_BALL_POS_Z[0], self.SPAWN_BALL_POS_Z[1]
            )
        (
            state_wrapper.ball.position[0],
            state_wrapper.ball.position[1],
        ) = fix_spawn_position(
            state_wrapper.ball.position[0], state_wrapper.ball.position[1]
        )

        # set ball vel
        state_wrapper.ball.linear_velocity[0] = random.uniform(
            self.SPAWN_BALL_VELOCITY[0], self.SPAWN_BALL_VELOCITY[1]
        )
        state_wrapper.ball.linear_velocity[1] = random.uniform(
            self.SPAWN_BALL_VELOCITY[0], self.SPAWN_BALL_VELOCITY[1]
        )
        state_wrapper.ball.linear_velocity[2] = random.uniform(
            self.SPAWN_BALL_VELOCITY[0], self.SPAWN_BALL_VELOCITY[1]
        )

        for car in state_wrapper.cars:
            pos = [0.0, 0.0, 0.0]
            yaw = 0.0
            pitch = 0.0
            roll = 0.0
            vel = [0.0, 0.0, 0.0]
            ang_vel = [0.0, 0.0, 0.0]

            # team_num = 0 = blue team
            if car.team_num == 0:
                # select a random spawn
                pos[0] = random.uniform(
                    self.SPAWN_BLUE_POS_X[0], self.SPAWN_BLUE_POS_X[1]
                )
                pos[1] = random.uniform(
                    self.SPAWN_BLUE_POS_Y[0], self.SPAWN_BLUE_POS_Y[1]
                )
                yaw = random.uniform(self.SPAWN_BLUE_YAW[0], self.SPAWN_BLUE_YAW[1])
                vel = [
                    random.uniform(
                        self.SPAWN_BLUE_VELOCITY_X[0],
                        self.SPAWN_BLUE_VELOCITY_X[1],
                    ),
                    random.uniform(
                        self.SPAWN_BLUE_VELOCITY_Y[0],
                        self.SPAWN_BLUE_VELOCITY_Y[1],
                    ),
                    random.uniform(
                        self.SPAWN_BLUE_VELOCITY_Z[0],
                        self.SPAWN_BLUE_VELOCITY_Z[1],
                    ),
                ]
                if floor_spawn:
                    pos[2] = 17
                    vel[2] = 0
                else:
                    pos[2] = random.uniform(
                        self.SPAWN_BLUE_POS_Z[0], self.SPAWN_BLUE_POS_Z[1]
                    )
                    pitch = random.uniform(
                        self.SPAWN_BLUE_PITCH[0], self.SPAWN_BLUE_PITCH[1]
                    )
                    roll = random.uniform(
                        self.SPAWN_BLUE_ROLL[0], self.SPAWN_BLUE_ROLL[1]
                    )
                    ang_vel = [
                        random.uniform(
                            self.SPAWN_BLUE_ANG_VELOCITY[0],
                            self.SPAWN_BLUE_ANG_VELOCITY[1],
                        ),
                        random.uniform(
                            self.SPAWN_BLUE_ANG_VELOCITY[0],
                            self.SPAWN_BLUE_ANG_VELOCITY[1],
                        ),
                        random.uniform(
                            self.SPAWN_BLUE_ANG_VELOCITY[0],
                            self.SPAWN_BLUE_ANG_VELOCITY[1],
                        ),
                    ]

            # team_num = 1 = orange team
            elif car.team_num == 1:
                # select a random spawn
                pos[0] = random.uniform(
                    self.SPAWN_ORANGE_POS_X[0], self.SPAWN_ORANGE_POS_X[1]
                )
                pos[1] = random.uniform(
                    self.SPAWN_ORANGE_POS_Y[0], self.SPAWN_ORANGE_POS_Y[1]
                )
                yaw = random.uniform(self.SPAWN_ORANGE_YAW[0], self.SPAWN_ORANGE_YAW[1])
                vel = [
                    random.uniform(
                        self.SPAWN_ORANGE_VELOCITY_X[0],
                        self.SPAWN_ORANGE_VELOCITY_X[1],
                    ),
                    random.uniform(
                        self.SPAWN_ORANGE_VELOCITY_Y[0],
                        self.SPAWN_ORANGE_VELOCITY_Y[1],
                    ),
                    random.uniform(
                        self.SPAWN_ORANGE_VELOCITY_Z[0],
                        self.SPAWN_ORANGE_VELOCITY_Z[1],
                    ),
                ]
                if floor_spawn:
                    pos[2] = 17
                    vel[2] = 0
                else:
                    pos[2] = random.uniform(
                        self.SPAWN_ORANGE_POS_Z[0], self.SPAWN_ORANGE_POS_Z[1]
                    )
                    pitch = random.uniform(
                        self.SPAWN_ORANGE_PITCH[0], self.SPAWN_ORANGE_PITCH[1]
                    )
                    roll = random.uniform(
                        self.SPAWN_ORANGE_ROLL[0], self.SPAWN_ORANGE_ROLL[1]
                    )
                    ang_vel = [
                        random.uniform(
                            self.SPAWN_ORANGE_ANG_VELOCITY[0],
                            self.SPAWN_ORANGE_ANG_VELOCITY[1],
                        ),
                        random.uniform(
                            self.SPAWN_ORANGE_ANG_VELOCITY[0],
                            self.SPAWN_ORANGE_ANG_VELOCITY[1],
                        ),
                        random.uniform(
                            self.SPAWN_ORANGE_ANG_VELOCITY[0],
                            self.SPAWN_ORANGE_ANG_VELOCITY[1],
                        ),
                    ]

            # set car state values
            pos[0], pos[1] = fix_spawn_position(pos[0], pos[1])
            car.set_pos(*pos)
            car.set_rot(yaw=yaw, pitch=pitch, roll=roll)
            car.boost = random.uniform(0.2, 0.8)
            car.set_lin_vel(*vel)
            car.set_ang_vel(*ang_vel)


class ShootState(StateSetter):
    # blue spawn
    SPAWN_BLUE_POS_X = [-SIDE_WALL_X + 256, SIDE_WALL_X - 256]
    SPAWN_BLUE_POS_Y = [-BACK_WALL_Y + 256, 500]
    SPAWN_BLUE_POS_Z = [17, 17]
    SPAWN_BLUE_YAW = [0.4 * np.pi, 0.6 * np.pi]
    SPAWN_BLUE_VELOCITY_X = [-50, 50]
    SPAWN_BLUE_VELOCITY_Y = [CAR_MAX_SPEED * 0.5, CAR_MAX_SPEED]
    SPAWN_BLUE_VELOCITY_Z = [0, 0]

    # Spawn orange near goal
    SPAWN_ORANGE_POS_X = [-SIDE_WALL_X + 256, SIDE_WALL_X - 256]
    SPAWN_ORANGE_POS_Y = [BACK_WALL_Y - 512, BACK_WALL_Y - 256]
    SPAWN_ORANGE_POS_Z = [17, 17]
    SPAWN_ORANGE_YAW = [-1 * np.pi, 1 * np.pi]
    SPAWN_ORANGE_VELOCITY_X = [-50, 50]
    SPAWN_ORANGE_VELOCITY_Y = [-50, 50]
    SPAWN_ORANGE_VELOCITY_Z = [0, 0]

    # Spawn ball relative to blue
    SPAWN_BALL_POS_X = [-50, 50]
    SPAWN_BALL_POS_Y = [800, 2000]
    SPAWN_BALL_POS_Z = [17, 17]
    SPAWN_BALL_VELOCITY = [-200, 200]

    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        """
        Modifies state_wrapper values to emulate a randomly selected attack scenario.

        :param state_wrapper: StateWrapper object to be modified with desired state values.
        """

        for car in state_wrapper.cars:
            pos = [0.0, 0.0, 0.0]
            yaw = 0.0
            vel = [0.0, 0.0, 0.0]

            # team_num = 0 = blue team
            if car.team_num == 0:
                # select a random spawn
                pos = [
                    random.uniform(self.SPAWN_BLUE_POS_X[0], self.SPAWN_BLUE_POS_X[1]),
                    random.uniform(self.SPAWN_BLUE_POS_Y[0], self.SPAWN_BLUE_POS_Y[1]),
                    random.uniform(self.SPAWN_BLUE_POS_Z[0], self.SPAWN_BLUE_POS_Z[1]),
                ]
                yaw = random.uniform(self.SPAWN_BLUE_YAW[0], self.SPAWN_BLUE_YAW[1])
                vel = [
                    random.uniform(
                        self.SPAWN_BLUE_VELOCITY_X[0],
                        self.SPAWN_BLUE_VELOCITY_X[1],
                    ),
                    random.uniform(
                        self.SPAWN_BLUE_VELOCITY_Y[0],
                        self.SPAWN_BLUE_VELOCITY_Y[1],
                    ),
                    random.uniform(
                        self.SPAWN_BLUE_VELOCITY_Z[0],
                        self.SPAWN_BLUE_VELOCITY_Z[1],
                    ),
                ]

                # set ball pos
                state_wrapper.ball.position = [
                    pos[0]
                    + random.uniform(
                        self.SPAWN_BALL_POS_X[0], self.SPAWN_BALL_POS_X[1]
                    ),
                    pos[1]
                    + random.uniform(
                        self.SPAWN_BALL_POS_Y[0], self.SPAWN_BALL_POS_Y[1]
                    ),
                    pos[2]
                    + random.uniform(
                        self.SPAWN_BALL_POS_Z[0], self.SPAWN_BALL_POS_Z[1]
                    ),
                ]
                (
                    state_wrapper.ball.position[0],
                    state_wrapper.ball.position[1],
                ) = fix_spawn_position(
                    state_wrapper.ball.position[0], state_wrapper.ball.position[1]
                )

            # team_num = 1 = orange team
            elif car.team_num == 1:
                # select a random spawn
                pos = [
                    random.uniform(
                        self.SPAWN_ORANGE_POS_X[0], self.SPAWN_ORANGE_POS_X[1]
                    ),
                    random.uniform(
                        self.SPAWN_ORANGE_POS_Y[0], self.SPAWN_ORANGE_POS_Y[1]
                    ),
                    random.uniform(
                        self.SPAWN_ORANGE_POS_Z[0], self.SPAWN_ORANGE_POS_Z[1]
                    ),
                ]
                yaw = random.uniform(self.SPAWN_ORANGE_YAW[0], self.SPAWN_ORANGE_YAW[1])
                vel = [
                    random.uniform(
                        self.SPAWN_ORANGE_VELOCITY_X[0],
                        self.SPAWN_ORANGE_VELOCITY_X[1],
                    ),
                    random.uniform(
                        self.SPAWN_ORANGE_VELOCITY_Y[0],
                        self.SPAWN_ORANGE_VELOCITY_Y[1],
                    ),
                    random.uniform(
                        self.SPAWN_ORANGE_VELOCITY_Z[0],
                        self.SPAWN_ORANGE_VELOCITY_Z[1],
                    ),
                ]

            # set ball vel
            state_wrapper.ball.linear_velocity = [
                random.uniform(
                    self.SPAWN_BALL_VELOCITY[0], self.SPAWN_BALL_VELOCITY[1]
                ),
                random.uniform(
                    self.SPAWN_BALL_VELOCITY[0], self.SPAWN_BALL_VELOCITY[1]
                ),
                random.uniform(
                    self.SPAWN_BALL_VELOCITY[0], self.SPAWN_BALL_VELOCITY[1]
                ),
            ]

            # set car state values
            pos[0], pos[1] = fix_spawn_position(pos[0], pos[1])
            car.set_pos(*pos)
            car.set_rot(yaw=yaw)
            car.boost = random.uniform(0.2, 0.8)
            car.set_lin_vel(*vel)


class DribbleState(StateSetter):
    # blue spawn
    SPAWN_BLUE_POS_X = [-SIDE_WALL_X + 256, SIDE_WALL_X - 256]
    SPAWN_BLUE_POS_Y = [-BACK_WALL_Y + 256, 500]
    SPAWN_BLUE_POS_Z = [17, 17]
    SPAWN_BLUE_YAW = [0.4 * np.pi, 0.6 * np.pi]
    SPAWN_BLUE_VELOCITY_X = [-50, 50]
    SPAWN_BLUE_VELOCITY_Y = [100, 0.5 * CAR_MAX_SPEED]
    SPAWN_BLUE_VELOCITY_Z = [0, 0]

    # Spawn orange near goal
    SPAWN_ORANGE_POS_X = [-SIDE_WALL_X + 256, SIDE_WALL_X - 256]
    SPAWN_ORANGE_POS_Y = [BACK_WALL_Y - 512, BACK_WALL_Y - 256]
    SPAWN_ORANGE_POS_Z = [17, 17]
    SPAWN_ORANGE_YAW = [-1 * np.pi, 1 * np.pi]
    SPAWN_ORANGE_VELOCITY_X = [-50, 50]
    SPAWN_ORANGE_VELOCITY_Y = [-50, 50]
    SPAWN_ORANGE_VELOCITY_Z = [0, 0]

    # Spawn ball relative to blue
    SPAWN_BALL_POS_X = [-5, 5]
    SPAWN_BALL_POS_Y = [-5, 5]
    SPAWN_BALL_POS_Z = [1.5 * BALL_RADIUS, 3 * BALL_RADIUS]
    SPAWN_BALL_VELOCITY = [-5, 5]

    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        """
        Modifies state_wrapper values to emulate a randomly selected attack scenario.

        :param state_wrapper: StateWrapper object to be modified with desired state values.
        """

        for car in state_wrapper.cars:
            pos = [0.0, 0.0, 0.0]
            yaw = 0.0
            vel = [0.0, 0.0, 0.0]

            # team_num = 0 = blue team
            if car.team_num == 0:
                # select a random spawn
                pos = [
                    random.uniform(self.SPAWN_BLUE_POS_X[0], self.SPAWN_BLUE_POS_X[1]),
                    random.uniform(self.SPAWN_BLUE_POS_Y[0], self.SPAWN_BLUE_POS_Y[1]),
                    random.uniform(self.SPAWN_BLUE_POS_Z[0], self.SPAWN_BLUE_POS_Z[1]),
                ]
                yaw = random.uniform(self.SPAWN_BLUE_YAW[0], self.SPAWN_BLUE_YAW[1])
                vel = [
                    random.uniform(
                        self.SPAWN_BLUE_VELOCITY_X[0],
                        self.SPAWN_BLUE_VELOCITY_X[1],
                    ),
                    random.uniform(
                        self.SPAWN_BLUE_VELOCITY_Y[0],
                        self.SPAWN_BLUE_VELOCITY_Y[1],
                    ),
                    random.uniform(
                        self.SPAWN_BLUE_VELOCITY_Z[0],
                        self.SPAWN_BLUE_VELOCITY_Z[1],
                    ),
                ]

                # set ball pos
                state_wrapper.ball.position = [
                    pos[0]
                    + random.uniform(
                        self.SPAWN_BALL_POS_X[0], self.SPAWN_BALL_POS_X[1]
                    ),
                    pos[1]
                    + random.uniform(
                        self.SPAWN_BALL_POS_Y[0], self.SPAWN_BALL_POS_Y[1]
                    ),
                    pos[2]
                    + random.uniform(
                        self.SPAWN_BALL_POS_Z[0], self.SPAWN_BALL_POS_Z[1]
                    ),
                ]
                (
                    state_wrapper.ball.position[0],
                    state_wrapper.ball.position[1],
                ) = fix_spawn_position(
                    state_wrapper.ball.position[0], state_wrapper.ball.position[1]
                )

                # set ball vel
                state_wrapper.ball.linear_velocity = [
                    vel[0]
                    + random.uniform(
                        self.SPAWN_BALL_VELOCITY[0], self.SPAWN_BALL_VELOCITY[1]
                    ),
                    vel[1]
                    + random.uniform(
                        self.SPAWN_BALL_VELOCITY[0], self.SPAWN_BALL_VELOCITY[1]
                    ),
                    vel[2]
                    + random.uniform(
                        self.SPAWN_BALL_VELOCITY[0], self.SPAWN_BALL_VELOCITY[1]
                    ),
                ]

            # team_num = 1 = orange team
            elif car.team_num == 1:
                # select a random spawn
                pos = [
                    random.uniform(
                        self.SPAWN_ORANGE_POS_X[0], self.SPAWN_ORANGE_POS_X[1]
                    ),
                    random.uniform(
                        self.SPAWN_ORANGE_POS_Y[0], self.SPAWN_ORANGE_POS_Y[1]
                    ),
                    random.uniform(
                        self.SPAWN_ORANGE_POS_Z[0], self.SPAWN_ORANGE_POS_Z[1]
                    ),
                ]
                yaw = random.uniform(self.SPAWN_ORANGE_YAW[0], self.SPAWN_ORANGE_YAW[1])
                vel = [
                    random.uniform(
                        self.SPAWN_ORANGE_VELOCITY_X[0],
                        self.SPAWN_ORANGE_VELOCITY_X[1],
                    ),
                    random.uniform(
                        self.SPAWN_ORANGE_VELOCITY_Y[0],
                        self.SPAWN_ORANGE_VELOCITY_Y[1],
                    ),
                    random.uniform(
                        self.SPAWN_ORANGE_VELOCITY_Z[0],
                        self.SPAWN_ORANGE_VELOCITY_Z[1],
                    ),
                ]

            # set car state values
            pos[0], pos[1] = fix_spawn_position(pos[0], pos[1])
            car.set_pos(*pos)
            car.set_rot(yaw=yaw)
            car.boost = random.uniform(0.2, 0.8)
            car.set_lin_vel(*vel)