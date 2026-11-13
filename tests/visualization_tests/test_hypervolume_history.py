from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pytest

from optuna.samplers import NSGAIISampler
from optuna.study import create_study
from optuna.trial import create_trial
from optuna.trial import Trial
from optuna.visualization._hypervolume_history import _get_hypervolume_history_info
from optuna.visualization._hypervolume_history import _HypervolumeHistoryInfo


@pytest.mark.parametrize(
    "directions",
    [
        ["minimize", "minimize"],
        ["minimize", "maximize"],
        ["maximize", "minimize"],
        ["maximize", "maximize"],
    ],
)
def test_get_optimization_history_info(
    directions: Sequence[Literal["minimize", "maximize"]],
) -> None:
    signs = [1 if d == "minimize" else -1 for d in directions]

    def objective(trial: Trial) -> Sequence[float]:
        def impl(trial: Trial) -> Sequence[float]:
            if trial.number == 0:
                return 1.5, 1.5  # dominated by the reference_point
            elif trial.number == 1:
                return 0.75, 0.75
            elif trial.number == 2:
                return 0.5, 0.5  # dominates Trial #1
            elif trial.number == 3:
                return 0.5, 0.5  # dominates Trial #1
            elif trial.number == 4:
                return 0.75, 0.25  # incomparable
            return 0.0, 0.0  # dominates all

        values = impl(trial)
        # Set constraint: trial 2 is infeasible
        if trial.number == 2:
            trial.set_constraint("constraint", 1.0)
        else:
            trial.set_constraint("constraint", 0.0)

        return signs[0] * values[0], signs[1] * values[1]

    sampler = NSGAIISampler()
    study = create_study(directions=directions, sampler=sampler)
    study.optimize(objective, n_trials=6)

    reference_point = np.asarray(signs)
    info = _get_hypervolume_history_info(study, reference_point)
    assert info == _HypervolumeHistoryInfo(
        trial_numbers=[0, 1, 2, 3, 4, 5], values=[0.0, 0.0625, 0.0625, 0.25, 0.3125, 1.0]
    )


@pytest.mark.parametrize("direction", ["minimize", "maximize"])
@pytest.mark.parametrize("infinite_reference", [False, True])
def test_hypervolume_history_preserves_infinite_volume(
    direction: Literal["minimize", "maximize"], infinite_reference: bool
) -> None:
    sign = 1 if direction == "minimize" else -1
    study = create_study(directions=[direction, direction])
    first_coordinate = 0.0 if infinite_reference else -float("inf")
    values = [[first_coordinate, 0.0], [first_coordinate, -1.0], [0.5, -2.0]]
    for value in values:
        study.add_trial(create_trial(values=[sign * x for x in value]))
    reference_point = sign * np.array([float("inf") if infinite_reference else 1.0, 1.0])

    with np.errstate(invalid="raise"):
        info = _get_hypervolume_history_info(study, reference_point)

    assert info == _HypervolumeHistoryInfo(trial_numbers=[0, 1, 2], values=[float("inf")] * 3)
