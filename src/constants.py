TICK_SKIP = 6
PHYSICS_TICKRATE = 120
HALF_LIFE_SECONDS = 5
TRAIN_DEVICE = "cpu"


def seconds_to_steps(seconds):
    return int(round(seconds * PHYSICS_TICKRATE / TICK_SKIP))


def seconds_to_frames(seconds):
    return int(round(seconds * PHYSICS_TICKRATE))


def frames_to_seconds(frames):
    return int(round(frames / PHYSICS_TICKRATE))
