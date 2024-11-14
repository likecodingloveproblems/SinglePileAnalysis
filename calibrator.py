from typing import Any, Callable

import numpy as np
from pydantic import BaseModel
from scipy.interpolate import interp1d
from scipy.optimize import differential_evolution

from fea_model import Pile, CalibrationParams


onill1982_single_pile_force_deformation = [
    [0, 0.45e-3, 1e-3, 1.4e-3, 2e-3, 2.85e-3, 3.4e-3, 4.2e-3],
    [0, 118e3, 250e3, 331e3, 436e3, 550e3, 600e3, 653e3],
]


class Calibrator(BaseModel):
    get_pile: Callable[[CalibrationParams], Pile]
    load_test_results: list
    load: float = None
    max_iter: int = 1000
    pop_size: int = 15

    def model_post_init(self, __context: Any) -> None:
        self.load = max(self.load_test_results[1])

    def calibrate(self):
        """Calibrate parameters to minimize the error of model with respect to pile load test"""
        de = differential_evolution(
            self.cost,
            bounds=CalibrationParams.bounds,
            maxiter=self.max_iter,
            popsize=self.pop_size,
        )
        final_calibration_params = CalibrationParams.from_array(de.x)
        return final_calibration_params

    def cost(self, args):
        calibration_params = CalibrationParams.from_array(args)
        pile = self.get_pile(calibration_params)
        pile_head_disp, pile_head_force = pile.analyze()
        pile_head_force_interp = interp1d(
            pile_head_disp, pile_head_force, kind="linear", fill_value="extrapolate"
        )
        pile_head_force = pile_head_force_interp(self.load_test_results[0])
        cost_ = self.least_square(pile_head_force, self.load_test_results[1])
        if np.isnan(cost_):
            return 1e6
        return cost_

    @staticmethod
    def least_square(y1, y2):
        return np.sqrt(sum((y2 - y1) ** 2))
