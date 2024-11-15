from typing import Any

import numpy as np
from pydantic import BaseModel
from scipy.interpolate import interp1d
from scipy.optimize import differential_evolution

from fea_model import CalibrationParams
from load_tests import LoadTestResult


class Calibrator(BaseModel):
    load_test_result: LoadTestResult
    load: float = None
    max_iter: int = 1000
    pop_size: int = 15

    def model_post_init(self, __context: Any) -> None:
        self.load = max(self.load_test_result.forces)

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
        pile = self.load_test_result.get_pile(calibration_params)
        pile_head_disp, pile_head_force = pile.analyze()
        pile_head_force_interp = interp1d(
            pile_head_disp, pile_head_force, kind="linear", fill_value="extrapolate"
        )
        pile_head_force = pile_head_force_interp(self.load_test_result.displacements)
        cost_ = self.least_square(pile_head_force, self.load_test_result.forces)
        if np.isnan(cost_):
            return 1e6
        return cost_

    @staticmethod
    def least_square(y1, y2):
        return np.sqrt(sum((y2 - y1) ** 2))
