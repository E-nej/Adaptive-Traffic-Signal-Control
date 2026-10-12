"""Seed ranges for generated scenarios (routes + SUMO's own RNG).

Training, validation, calibration and evaluation never share a scenario:

    training     100000·seed + episode   (seed 0..6, episode < 100000)
    validation   700001 + i
    acceptance   750000 + i              (learned policy vs its fallback, after training)
    calibration  800000 + i              (demand used to compute Webster plans)
    evaluation   900000 + i

With 100000·seed, training seed 7 would reach the validation range, so only
seeds 0..6 are allowed.  Widening the ranges would change every scenario of
the existing results, so it is left for a later version.
"""

TRAIN_STRIDE = 100000
MAX_TRAIN_SEED = 6
VALIDATION = 700001
ACCEPTANCE = 750000
CALIBRATION = 800000
EVALUATION = 900000


def check_train_seed(seed):
    if not 0 <= seed <= MAX_TRAIN_SEED:
        raise ValueError(f'Training seed must be in 0..{MAX_TRAIN_SEED} (got {seed}); larger seeds '
                         f'would reuse validation/calibration/evaluation scenarios')
    return seed


def train(seed, episode):
    check_train_seed(seed)
    if not 0 <= episode < TRAIN_STRIDE:
        raise ValueError(f'episode out of range: {episode}')
    return TRAIN_STRIDE * seed + episode


def validation(i):
    return VALIDATION + i


def acceptance(i):
    return ACCEPTANCE + i


def calibration(i):
    return CALIBRATION + i


def evaluation(i):
    return EVALUATION + i


def is_evaluation(seed):
    return seed >= EVALUATION
